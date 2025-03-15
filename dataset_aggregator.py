import datetime
import io
import os.path as osp
import random
import traceback
import copy
from collections import defaultdict

import dmc

import numpy as np
import multiprocessing as mp
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

from replay_buffer import load_episode, relabel_episode


# def episode_len(episode):
#     # subtract -1 because the dummy first transition
#     return next(iter(episode.values())).shape[0] - 1


# def save_episode(episode, fn):
#     with io.BytesIO() as bs:
#         np.savez_compressed(bs, **episode)
#         bs.seek(0)
#         with fn.open('wb') as f:
#             f.write(bs.read())


# def load_episode(fn):
#     with fn.open('rb') as f:
#         episode = np.load(f)
#         episode = {k: episode[k] for k in episode.keys()}
#         return episode


# def relabel_episode(env, episode):
#     rewards = []
#     reward_spec = env.reward_spec()
#     states = episode['physics']
#     for i in range(states.shape[0]):
#         with env.physics.reset_context():
#             env.physics.set_state(states[i])
#         reward = env.task.get_reward(env.physics)
#         reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
#         rewards.append(reward)
#     episode['reward'] = np.array(rewards, dtype=reward_spec.dtype)
#     return episode


class OfflineDatasetAggregator:
    def __init__(self, task, dataset_dir, skip_size, max_size, num_workers, relabel_reward):
        self._task = task
        self._dataset_dir = dataset_dir
        self._skip_size = skip_size
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._relabel_reward = relabel_reward

    def _worker_init_fn(self):
        worker_id = mp.current_process()._identity[0]

        seed = int(np.random.get_state()[1][0] + worker_id)
        np.random.seed(seed)
        random.seed(seed)

    def _worker_fn(self, args):
        eps_fns, relabel, task = args
        env = dmc.make(task)

        dataset = {}
        for eps_fn in eps_fns:
            # eps_idx, _ = [int(x) for x in eps_fn.stem.split('_')[1:]]
            # if eps_idx % self._num_workers != worker_id:
            #     continue

            episode = load_episode(eps_fn)
            if relabel:
                episode = relabel_episode(env, episode)

            for k in ['observation', 'action'] + (['reward'] if relabel else []):
                if k == 'observation':
                    episode_v = episode[k][:-1]
                else:
                    episode_v = episode[k][1:]
                if k + 's' not in dataset:
                    dataset[k + 's'] = episode_v
                else:
                    dataset[k + 's'] = np.concatenate([
                        dataset[k + 's'], episode_v], axis=0)

            if 'next_observations' not in dataset:
                dataset['next_observations'] = episode['observation'][1:]
            else:
                dataset['next_observations'] = np.concatenate([
                    dataset['next_observations'], episode['observation'][1:]], axis=0)

            if 'terminals' not in dataset:
                terminals = 1.0 - episode['discount'][1:]
                terminals[-1] = 1.0

                dataset['masks'] = 1.0 - terminals
                dataset['terminals'] = terminals
            else:
                terminals = 1.0 - episode['discount'][1:]
                terminals[-1] = 1.0
                dataset['masks'] = np.concatenate([
                    dataset['masks'], 1.0 - terminals], axis=0)
                dataset['terminals'] = np.concatenate([
                    dataset['terminals'], terminals], axis=0)

        return dataset

    def load(self):
        # def worker_fn(eps_fn):
        #     episode = load_episode(eps_fn)
        #     if relabel:
        #         episode = self._relabel_reward(episode)
        #
        #     return episode

        # observations=dataset['observations'].astype(np.float32),
        # actions=dataset['actions'].astype(np.float32),
        # next_observations=dataset['next_observations'].astype(np.float32),
        # terminals=terminals.astype(np.float32),
        # rewards=rewards
        # masks=masks

        # with mp.Manager() as manager:
        #     # episodes = manager.dict(observation=[], action=[], reward=[], discount=[])
        #     # episodes = manager.dict({
        #     #     'observation': manager.list(),
        #     #     'action': manager.list(),
        #     #     'reward': manager.list(),
        #     #     'discount': manager.list(),
        #     # })
        #     mp_dataset = manager.dict()
        #
        #     eps_fns = sorted(self._replay_dir.glob('*.npz'))[:200]
        #     split_size = int(np.ceil(len(eps_fns) / self._num_workers))
        #     worker_args = [(eps_fns[i:i + split_size], relabel, self._task)
        #                    for i in range(0, len(eps_fns), split_size)]
        #     with mp.Pool(processes=self._num_workers, initializer=self._worker_init_fn) as pool:
        #         datasets = pool.map(self._worker_fn, worker_args)
        #
        #     dataset = dict(mp_dataset)

        all_eps_fns = sorted(self._dataset_dir.glob('*.npz'))
        size, eps_fns = 0, []
        for eps_fn in all_eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            size += eps_len
            if size <= self._skip_size:
                continue
            elif size > (self._skip_size + self._max_size):
                break
            eps_fns.append(eps_fn)

        split_size = int(np.ceil(len(eps_fns) / self._num_workers))
        worker_args = [(eps_fns[i:i + split_size], self._relabel_reward, self._task)
                       for i in range(0, len(eps_fns), split_size)]
        with mp.Pool(processes=self._num_workers, initializer=self._worker_init_fn) as pool:
            results = pool.map(self._worker_fn, worker_args)

        # aggregate datasets
        dataset = dict()
        for result in results:
            for k, v in result.items():
                if k not in dataset:
                    dataset[k] = v.squeeze()
                else:
                    dataset[k] = np.concatenate([dataset[k], v.squeeze()], axis=0)

        return dataset


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_dataset(task, dataset_dir, skip_size, max_size, num_workers, relabel_reward):
    # max_size_per_worker = max_size // max(1, num_workers)

    aggregator = OfflineDatasetAggregator(task, dataset_dir, skip_size, max_size, num_workers, relabel_reward)
    if relabel_reward:
        print('Loading data...')
    else:
        print('Loading and labeling data...')
    dataset = aggregator.load()
    print('Dataset loaded.')

    return dataset

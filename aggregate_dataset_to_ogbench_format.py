import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import os.path as osp

from pathlib import Path

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'


import h5py
import hydra
import torch

import utils
from dataset_aggregator import make_dataset

torch.backends.cudnn.benchmark = True


def get_domain(task):
    if task.startswith('point_mass_maze'):
        return 'point_mass_maze'
    return task.split('_', 1)[0]


# def get_data_seed(seed, num_data_seeds):
#     return (seed - 1) % num_data_seeds + 1


# def eval(global_step, agent, env, logger, num_eval_episodes, video_recorder):
#     step, episode, total_reward = 0, 0, 0
#     eval_until_episode = utils.Until(num_eval_episodes)
#     while eval_until_episode(episode):
#         time_step = env.reset()
#         video_recorder.init(env, enabled=(episode == 0))
#         while not time_step.last():
#             with torch.no_grad(), utils.eval_mode(agent):
#                 action = agent.act(time_step.observation,
#                                    global_step,
#                                    eval_mode=True)
#             time_step = env.step(action)
#             video_recorder.record(env)
#             total_reward += time_step.reward
#             step += 1
#
#         episode += 1
#         video_recorder.save(f'{global_step}.mp4')
#
#     with logger.log_and_dump_ctx(global_step, ty='eval') as log:
#         log('episode_reward', total_reward / episode)
#         log('episode_length', step / episode)
#         log('step', global_step)

def save_dataset_to_h5(dataset, h5_path):
    with h5py.File(h5_path, 'w') as f:
        for k, v in dataset.items():
            f.create_dataset(k, data=v)


@hydra.main(config_path='.', config_name='aggregation_config')
def main(cfg):
    # work_dir = Path.cwd()
    # print(f'workspace: {work_dir}')

    utils.set_seed_everywhere(cfg.seed)
    # device = torch.device(cfg.device)

    # create logger
    # logger = Logger(work_dir, use_tb=cfg.use_tb)

    # create envs
    # env = dmc.make(cfg.task, seed=cfg.seed)

    # create agent
    # agent = hydra.utils.instantiate(cfg.agent,
    #                                 obs_shape=env.observation_spec().shape,
    #                                 action_shape=env.action_spec().shape)

    # create replay buffer
    # data_specs = (env.observation_spec(), env.action_spec(), env.reward_spec(),
    #               env.discount_spec())

    # create data storage
    domain = get_domain(cfg.task)
    datasets_dir = Path(cfg.datasets_dir).resolve()
    dataset_dir = datasets_dir / domain / cfg.expl_agent / 'buffer'
    print(f'dataset dir: {dataset_dir}')

    dataset = make_dataset(
        cfg.task, dataset_dir,
        cfg.skip_size, cfg.dataset_size,
        cfg.num_workers,
        cfg.relabel_reward
    )

    #
    # dataset = {
    #     'observations': [],
    #     'actions': [],
    #     'rewards': [],
    #     'discounts': [],
    #     'next_observations': []
    # }
    # for obs, action, reward, discount, next_obs in replay_loader:
    #     dataset['observations'].append(obs)
    #     dataset['actions'].append(action)
    #     dataset['rewards'].append(reward)
    #     dataset['discounts'].append(discount)
    #     dataset['next_observations'].append(next_obs)
    # for k, v in dataset.items():
    #     dataset[k] = torch.cat(v, dim=0).detach().cpu().numpy()

    save_path = Path(cfg.save_path).resolve()
    save_path.parents[0].mkdir(exist_ok=True)
    save_dataset_to_h5(dataset, save_path)

    print("Save dataset to: {}".format(save_path))


if __name__ == '__main__':
    main()

"""
To do a cross evaluation through different tasks for omnirobot
"""
import glob
import os
import json
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime

from rl_baselines.utils import WrapFrameStack, computeMeanReward, printGreen
from srl_zoo.utils import printRed
from rl_baselines import AlgoType
from rl_baselines.registry import registered_rl


def loadConfigAndSetup(log_dir):
    """
    load training variable from a pre-trained model
    :param log_dir: the path where the model is located,
    example: logs/sc2cc/OmnirobotEnv-v0/srl_combination/ppo2/19-05-07_11h32_39
    :return: train_args, algo_name, algo_class(stable_baselines.PPO2), srl_model_path, env_kwargs
    """
    algo_name = ""
    for algo in list(registered_rl.keys()):
        if algo in log_dir:
            algo_name = algo
            break
    algo_class, algo_type, _ = registered_rl[algo_name]
    if algo_type == AlgoType.OTHER:
        raise ValueError(algo_name + " is not supported for evaluation")

    env_globals = json.load(open(log_dir + "env_globals.json", 'r'))
    train_args = json.load(open(log_dir + "args.json", 'r'))
    env_kwargs = {
        "renders": False,
        "shape_reward": False,
        "action_joints": train_args["action_joints"],
        "is_discrete": not train_args["continuous_actions"],
        "random_target": train_args.get('random_target', False),
        "srl_model": train_args["srl_model"]
    }

    # load it, if it was defined
    if "action_repeat" in env_globals:
        env_kwargs["action_repeat"] = env_globals['action_repeat']

    # Remove up action
    if train_args["env"] == "Kuka2ButtonGymEnv-v0":
        env_kwargs["force_down"] = env_globals.get('force_down', True)
    else:
        env_kwargs["force_down"] = env_globals.get('force_down', False)

    if train_args["env"] == "OmnirobotEnv-v0":
        env_kwargs["simple_continual_target"] = env_globals.get(
            "simple_continual_target", False)
        env_kwargs["circular_continual_move"] = env_globals.get(
            "circular_continual_move", False)
        env_kwargs["square_continual_move"] = env_globals.get(
            "square_continual_move", False)
        env_kwargs["eight_continual_move"] = env_globals.get(
            "eight_continual_move", False)

    srl_model_path = None
    if train_args["srl_model"] != "raw_pixels":
        train_args["policy"] = "mlp"
        path = env_globals.get('srl_model_path')

        if path is not None:
            env_kwargs["use_srl"] = True
            # Check that the srl saved model exists on the disk
            assert os.path.isfile(
                env_globals['srl_model_path']), "{} does not exist".format(
                env_globals['srl_model_path'])
            srl_model_path = env_globals['srl_model_path']
            env_kwargs["srl_model_path"] = srl_model_path

    return train_args, algo_name, algo_class, srl_model_path, env_kwargs


def EnvsKwargs(task, env_kwargs):
    """
    create several environments kwargs
    :param tasks: the task we need the omnirobot to perform
    :param env_kwargs: the original env_kwargs from previous pre-trained odel
    :return: a list of env_kwargs that has the same length as tasks
    """
    t = task
    tmp = env_kwargs.copy()
    tmp['simple_continual_target'] = False
    tmp['circular_continual_move'] = False
    tmp['square_continual_move'] = False
    tmp['eight_continual_move'] = False

    if t == 'sc':
        tmp['simple_continual_target'] = True

    elif t == 'cc':
        tmp['circular_continual_move'] = True
    elif t == 'sqc':
        tmp['square_continual_move'] = True
    elif t == 'ec':
        tmp['eight_continual_move'] = True

    return tmp


def createEnv(
        model_dir,
        train_args,
        algo_name,
        algo_class,
        env_kwargs,
        log_dir="/tmp/gym/test/",
        num_cpu=1,
        seed=0):
    """
    create the environment from env)kwargs
    :param model_dir: The file name of the file which contains the pkl
    :param train_args:
    :param algo_name:
    :param algo_class:
    :param env_kwargs:
    :param log_dir:
    :param num_cpu:
    :param seed:
    :return:
    """
    # Log dir for testing the agent
    log_dir += "{}/{}/".format(algo_name,
                               datetime.now().strftime("%y-%m-%d_%Hh%M_%S_%f"))
    os.makedirs(log_dir, exist_ok=True)
    args = {
        "env": train_args['env'],
        "seed": seed,
        "num_cpu": num_cpu,
        "num_stack": train_args["num_stack"],
        "srl_model": train_args["srl_model"],
        "algo_type": train_args.get('algo_type', None),
        "log_dir": log_dir
    }
    # anonymous class so the dict looks like Arguments object
    algo_args = type('attrib_dict', (), args)()
    envs = algo_class.makeEnv(
        algo_args,
        env_kwargs=env_kwargs,
        load_path_normalise=model_dir)

    return log_dir, envs, algo_args


def policyEval(
        envs,
        model_path,
        log_dir,
        algo_class,
        algo_args,
        num_timesteps=251,
        num_cpu=1):
    """
    evaluation for the policy in the given envs
    :param envs: the environment we want to evaluate
    :param model_path: (str)the path to the policy ckp
    :param log_dir: (str) the path from a gym temporal file
    :param algo_class:
    :param algo_args:
    :param num_timesteps: (int) numbers of the timesteps we want to evaluate the policy
    :param num_cpu:
    :return:
    """
    tf.reset_default_graph()

    method = algo_class.load(model_path, args=algo_args)

    using_custom_vec_env = isinstance(envs, WrapFrameStack)

    obs = envs.reset()

    if using_custom_vec_env:
        obs = obs.reshape((1,) + obs.shape)
    n_done = 0
    last_n_done = 0
    episode_reward = []
    dones = [False for _ in range(num_cpu)]

    for i in range(num_timesteps):
        actions = method.getAction(obs, dones)
        obs, rewards, dones, _ = envs.step(actions)
        if using_custom_vec_env:
            obs = obs.reshape((1,) + obs.shape)
        if using_custom_vec_env:
            if dones:
                obs = envs.reset()
                obs = obs.reshape((1,) + obs.shape)

        n_done += np.sum(dones)
        if (n_done - last_n_done) > 1:
            last_n_done = n_done
            _, mean_reward = computeMeanReward(log_dir, n_done)
            episode_reward.append(mean_reward)
            printRed('Episode:{} Reward:{}'.format(n_done, mean_reward))
    _, mean_reward = computeMeanReward(log_dir, n_done)
    printRed('Episode:{} Reward:{}'.format(n_done, mean_reward))

    episode_reward.append(mean_reward)

    episode_reward = np.array(episode_reward)
    envs.close()
    return episode_reward


def latestPolicy(log_dir, algo_name):
    """
    Get the latest saved model from a file
    :param log_dir: (str) a path leads to the model saved path
    :param algo_name:
    :return: the file name of the latest saved policy and a flag
    """
    files = glob.glob(os.path.join(log_dir + algo_name + '_*_model.pkl'))
    files_list = []
    for file in files:
        eps = int((file.split('_')[-2]))
        files_list.append((eps, file))

    def sortFirst(val):
        return val[0]

    files_list.sort(key=sortFirst)
    if len(files_list) > 0:
        # episode,latest model file path, OK
        return files_list[-1][0], files_list[-1][1], True
    else:
        # No model saved yet
        return 0, '', False

import argparse
import subprocess
import re
import os
# Set the environment variable early, even before importing JAX
parser = argparse.ArgumentParser(description='Set GPU device for training.')
parser.add_argument('--gpu', type=str, help='GPU index to use', default=None)
args = parser.parse_args()

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPU: {args.gpu}")
else:
    # Function to get least used GPU if none specified
    def get_least_used_gpu():
        smi_output = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        gpu_memory = [re.split(r',\s*', line.strip()) for line in smi_output.stdout.strip().split('\n')]
        least_used_gpu = sorted(gpu_memory, key=lambda x: int(x[1]), reverse=True)[0][0]
        return least_used_gpu

    least_used_gpu = get_least_used_gpu()
    print(f"Didn't specify devices, using least used GPU: {least_used_gpu}")
    os.environ['CUDA_VISIBLE_DEVICES'] = least_used_gpu


import jax
import jax.numpy as jnp
import chex
import flax
import optax
from functools import partial
from typing import Dict, Sequence
import wandb
from logz.batch_logging import create_log_dict, batch_log
from craftax.craftax_env import make_craftax_env_from_name
import flashbax as fbx
from typing import Sequence, NamedTuple, Any

import numpy as np

from dreamerv3_flax.async_vector_env import AsyncVectorEnv
from dreamerv3_flax.buffer import ReplayBuffer
from dreamerv3_flax.env import CrafterEnv, VecCrafterEnv, TASKS
from dreamerv3_flax.jax_agent import JAXAgent

from dreamerv3_flax.craftax import CraftaxWrapper

@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def get_eval_metric(achievements: Sequence[Dict]) -> float:
    achievements = [list(achievement.values()) for achievement in achievements]
    success_rate = 100 * (np.array(achievements) > 0).mean(axis=0)
    score = np.exp(np.mean(np.log(1 + success_rate))) - 1
    eval_metric = {
        "success_rate": {k: v for k, v in zip(TASKS, success_rate)},
        "score": score,
    }
    return eval_metric


def main(config):

    # Seed
    np.random.seed(0)

    # Environment
    envs = CraftaxWrapper(config["NUM_ENVS"])
    # env_fns = [partial(CrafterEnv, seed=args.seed)]
    # env = VecCrafterEnv(AsyncVectorEnv(env_fns))

    # Buffer
    buffer = ReplayBuffer(envs, batch_size=16, num_steps=64)

    # Agent
    agent = JAXAgent(envs, seed=0)
    state = agent.initial_state(1)

    # Reset
    actions = envs.action_space.sample()
    obs, info = envs.reset()
    obs, rewards, terminateds, truncateds, infos = envs.step(actions)
    # obs["rgb"].shape = (4, 63, 63, 3)
    # rewards.shape = (4,)
    # terminateds.shape = (4,) # episode ends natuarally
    # truncateds.shape = (4,) # episode ends due to max episode length
    
    firsts = jnp.array([True for _ in range(config["NUM_ENVS"])])
    dones = jnp.array([False for _ in range(config["NUM_ENVS"])])

    # Train
    achievements = []
    for step in range(100000):
        print("Step:", step)
        actions, state = agent.act(obs["rgb"], firsts, state)
        
        buffer.add(obs["rgb"], actions, rewards, dones, firsts)
        
        firsts = dones

        actions = np.argmax(actions, axis=-1)
        obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        
        # if True in truncateds or True in terminateds, then set the corresponding element in dones to True
        dones = jnp.array([terminated or truncated for terminated, truncated in zip(terminateds, truncateds)])
        
        # breakpoint if there any true in truncateds or terminateds
        for i, done in enumerate(dones):
            if done and config["WANDB_MODE"] == "online":
                wandb.log(jax.tree.map(lambda x: x[i], infos), step)
        
        # if True in truncateds or True in terminateds:
        #     print("done for at least one env")
        #     breakpoint()
        #     metric = jax.tree.map(lambda x: (x * infos["returned_episode"]).sum() / infos["returned_episode"].sum(), infos)
        
        #     if config["WANDB_MODE"] == "online":
        #         wandb.log(metric, step)
        #         # def callback(metric, update_step):
        #         #     to_log = create_log_dict(metric, config)
        #         #     batch_log(update_step, to_log, config)

        #         # jax.debug.callback(callback, metric, step)
        #     # report on wandb if required
            
        # for done, info in zip(dones, infos):
        #     # print("breakpoint zip")
        #     # breakpoint()
        #     if done:
        #         rollout_metric = {
        #             "episode_return": info["returned_episode_returns"].item(),
        #             "episode_length": info["returned_episode_lengths"].item(),
        #             "time_step": info["timestep"].item(),
        #         }
        #         print("rollout_metric when done: ", rollout_metric)
        #         # print("breakpoint because of done")
        #         # breakpoint()
        #         # logger.log(rollout_metric, step)
        #         # achievements.append(info["achievements"])
        #         # eval_metric = get_eval_metric(achievements)
        #         # print("eval_metric when done: ", eval_metric)
        #         # logger.log(eval_metric, step)
        
        #         # Add an indented block here
        #         # Example:
        #         # if eval_metric["score"] > 0.8:
        #         #     print("High score!")

        if step >= 1024 and step % 2 == 0:
            data = buffer.sample()
            _, train_metric = agent.train(data)
            if step % 100 == 0 and config["WANDB_MODE"] == "online":
                wandb.log(train_metric, step)
                # jax.debug.callback(callback, train_metric, step)
                # print("Step:", step, "train_metric:", train_metric)
                # print(infos)
                # print("breakpoint because of step % 100 == 0")
                # breakpoint()
                # logger.log(train_metric, step)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--exp_name", type=str, required=True)
    # parser.add_argument("--timestamp", default=None, type=str)
    # parser.add_argument("--seed", default=0, type=int)
    # args = parser.parse_args()
    config = {
        "NUM_ENVS": int(10),
        "NUM_REPEATS": int(1),
        "BUFFER_SIZE": int(1e6),
        "BUFFER_BATCH_SIZE": int(128),
        "TOTAL_TIMESTEPS": 5e5,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 25e4,
        "TARGET_UPDATE_INTERVAL": 500,
        "LR": 2.5e-4,
        "LEARNING_STARTS": 10000,
        "TRAINING_INTERVAL": 10,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": "Craftax-Classic-Pixels-v1",
        # "ENV_NAME": "Craftax-Classic-Symbolic-v1",
        "SEED": 0,
        "NUM_SEEDS": 1,
        "WANDB_MODE": "online",  # set to online to activate wandb
        "ENTITY": "",
        "PROJECT": "dreamerv3_flax_craftax",
        "DEBUG": False,
    }
    
    if config["WANDB_MODE"] == "online":
        wandb.init(
            project=config["PROJECT"],
            config=config,
            name=config["ENV_NAME"]
            + "-dreamer_v3_flax-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )
    
    main(config)

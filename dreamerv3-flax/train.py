import argparse
from datetime import datetime
import subprocess
import re
import os
import orbax
import orbax.checkpoint
import shutil
import glob

from matplotlib import pyplot as plt

# Set the environment variable early, even before importing JAX
parser = argparse.ArgumentParser(description="Set GPU device for training.")
parser.add_argument("--gpu", type=str, help="GPU index to use", default=None)
args = parser.parse_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU: {args.gpu}")
else:
    # Function to get least used GPU if none specified
    def get_least_used_gpu():
        smi_output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
        )
        gpu_memory = [
            re.split(r",\s*", line.strip())
            for line in smi_output.stdout.strip().split("\n")
        ]
        least_used_gpu = sorted(gpu_memory, key=lambda x: int(x[1]), reverse=True)[0][0]
        return least_used_gpu

    least_used_gpu = get_least_used_gpu()
    print(f"Didn't specify devices, using least used GPU: {least_used_gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = least_used_gpu


import jax
import jax.numpy as jnp
import chex
import flax
import optax
from functools import partial
from typing import Dict, Sequence
import wandb

# from logz.batch_logging import create_log_dict, batch_log
from craftax.craftax_env import make_craftax_env_from_name
from typing import Sequence, NamedTuple, Any

import numpy as np

from dreamerv3_flax.async_vector_env import AsyncVectorEnv
from dreamerv3_flax.buffer import ReplayBuffer

# from dreamerv3_flax.env import CrafterEnv, VecCrafterEnv, TASKS
from dreamerv3_flax.jax_agent import JAXAgent

from dreamerv3_flax.craftax import CraftaxWrapper
from flax.training import orbax_utils


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


def save_checkpoint(
    step: int,
    agent: JAXAgent,
    buffer: ReplayBuffer,
    achievements: Sequence[Dict],
    config: Dict,
):
    state_dict = {
        "step": step,
        "agent_model_state": agent.model_state,
        "agent_policy_state": agent.policy_state,
        "buffer": {
            "obs": buffer.obs,
            "actions": buffer.actions,
            "rewards": buffer.rewards,
            "dones": buffer.dones,
            "firsts": buffer.firsts,
            "pos": buffer.pos,
            "full": buffer.full,
        },
        "achievements": achievements,
    }

    checkpoint_path = f"/home/alan/Craftax_Baselines/dreamerv3-flax/{config['timestamp']}_ckpt_seed_{config['SEED']}_step_{step}"
    save_args = orbax_utils.save_args_from_target(state_dict)
    orbax_checkpointer.save(
        checkpoint_path, state_dict, save_args=save_args, force=True
    )


def delete_previous_checkpoints(step: int, currenttime: str):
    search_pattern = f"/home/alan/Craftax_Baselines/dreamerv3-flax/{currenttime}_ckpt_seed_{config['SEED']}_step_{step}"
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:  # Check if the list is empty
        print("Nothing to delete")
        return False
    else:
        for file in list_of_files:
            if os.path.isdir(file):
                shutil.rmtree(file)
                print(f"Deleted directory: {file}")
            else:
                os.remove(file)
                print(f"Deleted file: {file}")
        return True


def load_latest_checkpoint(buffer: ReplayBuffer, agent: JAXAgent, config: Dict):
    step = 0
    achievements = []

    buffer_dict = {
        "obs": np.zeros_like(buffer.obs),
        "actions": np.zeros_like(buffer.actions),
        "rewards": np.zeros_like(buffer.rewards),
        "dones": np.zeros_like(buffer.dones),
        "firsts": np.zeros_like(buffer.firsts),
        "pos": 0,
        "full": False,
    }
    target = {
        "step": step,
        "agent_model_state": agent.model_state,
        "agent_policy_state": agent.policy_state,
        "buffer": buffer_dict,
        "achievements": achievements,
    }
    if not config["ckpt_filepath"]:
        # raise error
        raise ValueError("No checkpoint file path provided")
    else:
        print(f"Loading checkpoint from {config['ckpt_filepath']}")
        return orbax_checkpointer.restore(config["ckpt_filepath"], item=target)


def main(config):
    # Seed
    np.random.seed(config["SEED"])

    # Environment
    envs = CraftaxWrapper(config["NUM_ENVS"])
    # env_fns = [partial(CrafterEnv, seed=args.seed)]
    # env = VecCrafterEnv(AsyncVectorEnv(env_fns))

    # Buffer
    buffer = ReplayBuffer(
        envs, batch_size=16, num_steps=64, buffer_size=config["BUFFER_SIZE"]
    )

    # Agent
    agent = JAXAgent(envs, seed=config["SEED"])
    state = agent.initial_state(1)

    if config["ckpt_filepath"] and config["load_checkpoint"]:
        state_restored = load_latest_checkpoint(buffer, agent, config=config)
        if state_restored:
            step, agent.model_state, agent.policy_state, buffer_dict, achievements = (
                state_restored["step"],
                state_restored["agent_model_state"],
                state_restored["agent_policy_state"],
                state_restored["buffer"],
                state_restored["achievements"],
            )

            (
                buffer.obs,
                buffer.actions,
                buffer.rewards,
                buffer.dones,
                buffer.firsts,
                buffer.pos,
                buffer.full,
            ) = (
                buffer_dict["obs"].copy(),
                buffer_dict["actions"].copy(),
                buffer_dict["rewards"].copy(),
                buffer_dict["dones"].copy(),
                buffer_dict["firsts"].copy(),
                buffer_dict["pos"],
                buffer_dict["full"],
            )
    else:
        print("Starting from scratch")
        step = 0
        achievements = []

    # Reset
    actions = envs.action_space.sample()
    obs, info = envs.reset()
    obs, rewards, terminateds, truncateds, dones, infos = envs.step(actions)
    # obs["rgb"].shape = (4, 63, 63, 3)
    # rewards.shape = (4,)
    # terminateds.shape = (4,) # episode ends natuarally
    # truncateds.shape = (4,) # episode ends due to max episode length

    firsts = jnp.array([True for _ in range(config["NUM_ENVS"])])

    # Train
    achievements = []

    # from tqdm import tqdm

    # for step in tqdm(range(1, 1000001)):
    for step in range(1, 1000001):
        # actions, state = agent.act(obs["rgb"], firsts, state)
        # actions = envs.action_space.sample()

        # generate a numpy array randomly sample from int 1, 2, 3, 4 (left, right, up, down)
        actions = np.random.randint(1, 5, size=(config["NUM_ENVS"],))

        buffer.add(obs["rgb"], actions, rewards, dones, firsts)

        firsts = dones

        # actions = np.argmax(actions, axis=-1)
        obs, rewards, terminateds, truncateds, dones, infos = envs.step(actions)

        # breakpoint if there any true in truncateds or terminateds
        for i, done in enumerate(dones):
            if done and config["WANDB_MODE"] == "online":
                wandb.log(jax.tree.map(lambda x: x[i], infos), step)

        if step >= 1024 and step % 2 == 0:
            data = buffer.sample()
            _, train_metric = agent.train(data)
            if step % 100 == 0 and config["WANDB_MODE"] == "online":
                wandb.log(train_metric, step)

        if step % config["save_every"] == 0 and step > 0:
            if config["save_checkpoint"]:
                print(f"Step: {step}", "saving checkpoint")
                save_checkpoint(step, agent, buffer, achievements, config)
                print("Checkpoint saved")
                deleted = delete_previous_checkpoints(
                    step - config["save_every"], config["timestamp"]
                )
                if deleted:
                    print("Previous checkpoints deleted")
            else:
                print("Not saving checkpoints, Step: ", step)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--exp_name", type=str, required=True)
    # parser.add_argument("--timestamp", default=None, type=str)
    # parser.add_argument("--seed", default=0, type=int)
    # args = parser.parse_args()
    config = {
        "NUM_ENVS": int(1),
        "NUM_REPEATS": int(1),
        # "BUFFER_SIZE": int(1e6),
        "BUFFER_SIZE": int(5e5),
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
        "ckpt_filepath": None,
        "load_checkpoint": False,
        "save_checkpoint": True,
        "save_every": 100000,
    }
    config["timestamp"] = datetime.now().strftime("%Y%m%d-%H%M%S")

    if config["WANDB_MODE"] == "online":
        wandb.init(
            project=config["PROJECT"],
            config=config,
            name="-dv3_flax-s" + str(config["SEED"]),
        )

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    main(config)

import argparse
import os
import glob
import re
import subprocess
import numpy as np
import jax.numpy as jnp
import wandb
import dill as pickle  # Use dill instead of pickle
import orbax
import orbax.checkpoint
from functools import partial
from typing import Dict, Sequence
from dreamerv3_flax.async_vector_env import AsyncVectorEnv
from dreamerv3_flax.buffer import ReplayBuffer
from dreamerv3_flax.env import CrafterEnv, VecCrafterEnv, TASKS
from dreamerv3_flax.jax_agent import JAXAgent
from dreamerv3_flax.craftax import CraftaxWrapper
from flax.training import orbax_utils

def get_least_used_gpus(n=1):
    smi_output = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], capture_output=True, text=True)
    gpu_memory = [re.split(r',\s*', line.strip()) for line in smi_output.stdout.strip().split('\n')]
    least_used_gpus = sorted(gpu_memory, key=lambda x: int(x[1]), reverse=True)[:n]
    return [gpu[0] for gpu in least_used_gpus]

least_used_gpus = get_least_used_gpus(1)
print(f"Didn't specify devices, using least used GPUs: {', '.join(least_used_gpus)}")
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(least_used_gpus)

def get_eval_metric(achievements: Sequence[Dict]) -> float:
    achievements = [list(achievement.values()) for achievement in achievements]
    success_rate = 100 * (np.array(achievements) > 0).mean(axis=0)
    score = np.exp(np.mean(np.log(1 + success_rate))) - 1
    eval_metric = {
        "success_rate": {k: v for k, v in zip(TASKS, success_rate)},
        "score": score,
    }
    return eval_metric

def save_checkpoint(step: int, agent: JAXAgent, buffer: ReplayBuffer, achievements: Sequence[Dict]):
    state_dict = {
        'step': step,
        'agent_model_state': agent.model_state,
        'agent_policy_state': agent.policy_state,
        'buffer': {'obs': buffer.obs, 'actions': buffer.actions, 'rewards': buffer.rewards, 'dones': buffer.dones, 'firsts': buffer.firsts, 'pos': buffer.pos, 'full': buffer.full},
        'achievements': achievements
    }
    
    checkpoint_path = f"/home/alan/Craftax_Baselines/dreamerv3-flax/dreamerv3_flax/ckpt_seed_{args.seed}_step_{step}"
    save_args = orbax_utils.save_args_from_target(state_dict)
    orbax_checkpointer.save(checkpoint_path, state_dict, save_args=save_args, force=True)

def load_latest_checkpoint(list_of_files, buffer: ReplayBuffer, agent: JAXAgent):
    step = 0
    achievements = []
    buffer_dict = {'obs': np.zeros_like(buffer.obs), 'actions': np.zeros_like(buffer.actions), 'rewards': np.zeros_like(buffer.rewards), 'dones': np.zeros_like(buffer.dones), 'firsts': np.zeros_like(buffer.firsts), 'pos': 0, 'full': False}
    target = {
        'step': step,
        'agent_model_state': agent.model_state,
        'agent_policy_state': agent.policy_state,
        'buffer': buffer_dict,
        'achievements': achievements
    }
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading checkpoint from {latest_file}")
    return orbax_checkpointer.restore(latest_file, item=target)

def main(config, args):

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
    
    search_pattern = f"/home/alan/Craftax_Baselines/dreamerv3-flax/dreamerv3_flax/ckpt_seed_{args.seed}_step_*"
    list_of_files = glob.glob(search_pattern)
    print("List of files: ", list_of_files)
    if list_of_files and args.load_checkpoint:
        print("Loading checkpoint from /home/alan/Craftax_Baselines/dreamerv3-flax/dreamerv3_flax/ckpt_seed_"+str(args.seed))
        state_restored = load_latest_checkpoint(list_of_files, buffer, agent)
        if state_restored:
            step, agent.model_state, agent.policy_state, buffer_dict, achievements = (
                state_restored['step'],
                state_restored['agent_model_state'],
                state_restored['agent_policy_state'],
                state_restored['buffer'],
                state_restored['achievements']
            )
            buffer.obs, buffer.actions, buffer.rewards, buffer.dones, buffer.firsts, buffer.pos, buffer.full = (
                buffer_dict['obs'].copy(),
                buffer_dict['actions'].copy(),
                buffer_dict['rewards'].copy(),
                buffer_dict['dones'].copy(),
                buffer_dict['firsts'].copy(),
                buffer_dict['pos'],
                buffer_dict['full']
            )
    else:
        step = 0
        achievements = []

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
    for step in range(1000000):
        if step % 1000 == 0:
            print("Step: ", step)
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
                achievements.append({k.split('/')[1]: int(v[0]) for k, v in infos.items() if 'Achievements/' in k})
                eval_metric = get_eval_metric(achievements)
                wandb.log(eval_metric, step)
        

        if step >= 1024 and step % 2 == 0:
            data = buffer.sample()
            _, train_metric = agent.train(data)
            if step % 100 == 0 and config["WANDB_MODE"] == "online":
                wandb.log(train_metric, step)
                
        if step % 100000 == 0 and step > 0:
            print(f"Step: {step}", "saving checkpoint")
            save_checkpoint(step, agent, buffer, achievements)
            print("Checkpoint saved")


if __name__ == "__main__":
    config = {
        "NUM_ENVS": int(1),
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
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--load_checkpoint", default=False, action="store_true")
    args = parser.parse_args()
        
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    
    
    main(config, args)

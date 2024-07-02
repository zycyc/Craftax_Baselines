from __future__ import annotations

from jax.tree_util import tree_map

# from functools import partial
# from typing import Optional, Dict, Any, List, Tuple, Union, SupportsFloat
# from gymnax.environments.environment import Environment, EnvParams
# from gymnax.environments.spaces import gymnax_space_to_gym_space
# # from gymnax.environments import spaces
# from gymnasium import spaces
# from gymnax.wrappers.gym import GymnaxToVectorGymWrapper
# from craftax.craftax_env import make_craftax_env_from_name
# import gymnasium as gym
# import gymnax
# import jax
# import jax.numpy as jnp
# import jax.dlpack
# import torch
# import chex
# import numpy as np
# from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.functional import FuncEnv
from gymnasium.envs.functional_jax_env import FunctionalJaxVectorEnv
from craftax.craftax_env import make_craftax_env_from_name
from gymnasium import spaces
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import craftax
from typing import Any
import chex
import jax.random as jrng
from dataclasses import asdict
from wrappers import LogWrapper

@dataclass
class MegaState:
    craftax_state: craftax.craftax_classic.envs.craftax_state.EnvState
    observation: jnp.ndarray
    terminal: bool
    reward: jnp.ndarray
    info: dict

    # Need this to make the state subscriptable
    @property
    def at(self):
        return _MegaStateAt(self, ())

class _MegaStateAt:
    def __init__(self, state, idx):
        self.state = state
        self.idx = idx
    
    def __getitem__(self, idx):
        return _MegaStateAt(self.state, idx)

    def set(self, new_state):
        return _set_at(self.state, self.idx, new_state)

def _set_at(state, idx, new_state):
    def set_leaf(leaf, new_leaf):
        if isinstance(leaf, jnp.ndarray):
            return leaf.at[idx].set(new_leaf)
        elif isinstance(leaf, list):
            leaf = list(leaf)  # Ensure it's mutable
            leaf[idx] = new_leaf
            return leaf
        elif isinstance(leaf, dict):
            leaf = dict(leaf)  # Ensure it's mutable
            leaf[idx] = new_leaf
            return leaf
        else:
            return new_leaf

    return tree_map(set_leaf, state, new_state)



chex.register_dataclass_type_with_jax_tree_util(MegaState)
class CraftaxFuncEnv(FuncEnv):
    def __init__(self):
        super().__init__()
        self.env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=True)
        self.env = LogWrapper(self.env)
        params = self.env.default_params
        self.observation_space = spaces.Dict({"rgb": spaces.Box(0, 255, self.env.observation_space(params).shape, dtype=jnp.uint8)})
        self.action_space = spaces.Discrete(17)
        self.params = params
    
    def initial(self, rng):
        obs, state = self.env.reset(rng)
        obs = {"rgb": obs}
        return MegaState(state, obs, False, jnp.array(0.0), {})

    def transition(self, state, action, rng):
        obs, env_state, reward, done, info = self.env.step(rng, state.craftax_state, action, self.params)
        obs = {"rgb": obs}
        return MegaState(env_state, obs, done, reward, info)

    def observation(self, state):
        return state.observation # To conform to the gymnasium wrapper api

    def reward(self, state, action, next_state):
        return next_state.reward

    def terminal(self, state):
        return state.terminal
    
    def state_info(self, state: Any, params: Any | None = None) -> dict:
        infos = []
        for env_id in range(state.reward.val.shape[0]):
            info = {}
            for key, value in state.info.items():
                info[key] = value[env_id]
            infos.append(info)
        return infos
    
    def transition_info(self, state, action, next_state, params: Any | None = None) -> dict:
        infos = []
        for env_id in range(state.reward.shape[0]):
            info = {}
            for key, value in state.info.items():
                info[key] = value[env_id]
            infos.append(info)
        return infos


class CraftaxWrapper(FunctionalJaxVectorEnv):
    def __init__(self, 
                num_envs: int):
    
        craftax_env = CraftaxFuncEnv()
        metadata = {"jax": True}
        super().__init__(craftax_env, num_envs, max_episode_steps=200, metadata=metadata, render_mode="rgb")
        
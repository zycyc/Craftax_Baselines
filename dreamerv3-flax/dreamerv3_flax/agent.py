from __future__ import annotations
from functools import partial
from typing import Sequence

from chex import Array, ArrayTree
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from dreamerv3_flax.policy import Policy
from dreamerv3_flax.world_model import WorldModel


class Agent(nn.Module):
    """Agent module."""

    obs_shape: Sequence[int]
    num_actions: int
    img_horizon: int = 15

    def setup(self):
        """Initializes an agent."""
        # Model
        self.model = WorldModel(self.obs_shape, self.num_actions)

        # Policy
        self.policy = Policy(self.num_actions)

    def initial_state(self, batch_size: int) -> ArrayTree:
        """Returns the initial state."""
        return self.model.initial_state(batch_size)

    def latent_size(self) -> int:
        """Returns the latent size."""
        return self.model.rssm.latent_size()

    def model_loss(
        self,
        obs: Array,
        action: Array,
        reward: Array,
        cont: Array,
        first: Array,
        state: ArrayTree | None = None,
    ) -> ArrayTree:
        """Calculates the model loss."""
        return self.model.model_loss(obs, action, reward, cont, first, state)

    def policy_loss(
        self,
        latent: Array,
        action: Array,
        reward: Array,
        cont: Array,
    ) -> ArrayTree:
        """Calculates the policy loss."""
        return self.policy.policy_loss(latent, action, reward, cont)

    def update_policy(self):
        """Updates the policy."""
        self.policy.update_policy()

    def act(
        self,
        obs: Array,
        first: Array,
        state: ArrayTree | None = None,
    ) -> ArrayTree:
        """Samples an action."""
        # Get the initial state if none.
        if state is None:
            batch_size = first.shape[0]
            state = self.initial_state(batch_size)

        # Transform the observation.
        obs = jnp.astype(obs, jnp.float32) / 255.0

        # Encode the observation.
        encoded = self.model.encoder(obs)

        # Run the RSSM observation step.
        post, _ = self.model.rssm.obs_step(*state, encoded, first)

        # Sample an action.
        latent = self.model.rssm.get_latent(post)
        action = self.policy.act(latent)

        # Update the state.
        state = (post, action)

        return action, state

    @staticmethod
    def img_step(agent: Agent, state: ArrayTree, sample: bool = True) -> ArrayTree:
        """Runs an imagination step."""
        # Run a RSSM imagination step.
        prior = agent.model.rssm.img_step(*state, sample=sample)

        # Sample an action.
        latent = agent.model.rssm.get_latent(prior)
        action = agent.policy.act(latent)

        # Update the state.
        state = (prior, action)

        return state, (latent, action)

    @staticmethod
    def img_step_4_rnd_dir_act(
        agent: Agent, state: ArrayTree, key, sample: bool = True
    ) -> ArrayTree:
        """Runs an imagination step but with random actions (left, right, up, down)."""
        # Run a RSSM imagination step.
        prior = agent.model.rssm.img_step(*state, sample=sample)

        # Sample an action.
        latent = agent.model.rssm.get_latent(prior)
        action = jax.random.randint(minval=1, maxval=5, shape=(1,), key=key)
        # turn into one-hot (17 dim)
        action = jax.nn.one_hot(action, num_classes=17).squeeze()

        # Update the state.
        state = (prior, action)

        return state, (latent, action)

    def imagine(self, post: ArrayTree, cont: Array, **kwargs) -> ArrayTree:
        """Runs an imagination."""

        # Flatten the posterior state and continuation.
        def flatten(x: Array) -> Array:
            return jnp.reshape(x, (-1, *x.shape[2:]))

        post = tree_map(flatten, post)
        cont = flatten(cont)

        # Define the input RSSM state and continuation.
        state_in = post
        cont_in = cont

        # Define the input state.
        latent_in = self.model.rssm.get_latent(state_in)
        action_in = self.policy.act(latent_in)
        state_in = (state_in, action_in)

        # Run an imagination step.
        scan = nn.scan(
            self.img_step,
            variable_broadcast=["params", "stats"],
            split_rngs={"params": False, "prior": True, "action": True},
            in_axes=0,
            out_axes=0,
            length=self.img_horizon,
        )
        _, (latent, action) = scan(self, state_in)

        # Calculate the reward and continuation.
        reward = self.model.get_reward(latent)
        cont = self.model.get_cont(latent)

        # Concatenate the input and imagined data.
        latent = jnp.concatenate([latent_in[None], latent], axis=0)
        action = jnp.concatenate([action_in[None], action], axis=0)
        cont = jnp.concatenate([cont_in[None], cont], axis=0)

        # Define the trajectory.
        traj = {"latent": latent, "action": action, "reward": reward, "cont": cont}

        return traj

    def simple_imagine(
        self, post: ArrayTree, sample: bool = False, **kwargs
    ) -> ArrayTree:
        """Runs a simple imagination for just latents."""

        # Flatten the posterior state and continuation.
        def flatten(x: Array) -> Array:
            return jnp.reshape(x, (-1, *x.shape[2:]))

        post = tree_map(flatten, post)

        # Define the input RSSM state and continuation.
        state_in = post

        # Define the input state.
        latent_in = self.model.rssm.get_latent(state_in)
        action_in = self.policy.act(latent_in)
        state_in = (state_in, action_in)

        # Run an imagination step.
        scan_fn = partial(self.img_step, sample=sample)
        scan = nn.scan(
            scan_fn,
            variable_broadcast=["params", "stats"],
            split_rngs={"params": False, "prior": True, "action": True},
            in_axes=0,
            out_axes=0,
            length=self.img_horizon,
        )
        _, (latent, action) = scan(self, state_in)

        # Calculate the reward and continuation.
        reward = self.model.get_reward(latent)

        # Concatenate the input and imagined data.
        latent = jnp.concatenate([latent_in[None], latent], axis=0)
        action = jnp.concatenate([action_in[None], action], axis=0)

        # Define the trajectory.
        traj = {"latent": latent, "action": action, "reward": reward}

        return traj

    def simple_imagine_4_rnd_dir_act(
        self, post: ArrayTree, key, sample: bool = False, **kwargs
    ) -> ArrayTree:
        """Runs a simple imagination for just latents."""

        # Flatten the posterior state and continuation.
        def flatten(x: Array) -> Array:
            return jnp.reshape(x, (-1, *x.shape[2:]))

        post = tree_map(flatten, post)

        # Define the input RSSM state and continuation.
        state_in = post

        # Define the input state.
        latent_in = self.model.rssm.get_latent(state_in)
        # action_in = self.policy.act(latent_in)
        key, _key = jax.random.split(key)
        action_in = jax.random.randint(minval=1, maxval=5, shape=(1,), key=_key)
        action_in = jax.nn.one_hot(action_in, num_classes=17).squeeze()
        state_in = (state_in, action_in)

        # Run an imagination step.
        scan_fn = partial(self.img_step_4_rnd_dir_act, sample=sample)
        key = jax.random.split(key, self.img_horizon)
        scan = nn.scan(
            scan_fn,
            variable_broadcast=["params", "stats"],
            split_rngs={"params": False, "prior": True, "action": True},
            in_axes=0,
            out_axes=0,
            length=self.img_horizon,
        )
        _, (latent, action) = scan(self, state_in, key)

        # Calculate the reward and continuation.
        reward = self.model.get_reward(latent)

        # Concatenate the input and imagined data.
        latent = jnp.concatenate([latent_in[None], latent], axis=0)
        action = jnp.concatenate([action_in[None], action], axis=0)

        # Define the trajectory.
        traj = {"latent": latent, "action": action, "reward": reward}

        return traj

    def simple_decode(self, latent: ArrayTree) -> Array:
        """Decodes a latent."""
        return self.model.decoder(latent).mode()

__doc__ = """ File for training algorithms.

The main base class is `Trainer`, and algorithms are implemented as subclasses.
See the `Trainer` class for details."""

import abc
import copy
import random
from abc import ABC
from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from callbacks import Callback


def choose_action_greedy(
    state: Union[np.ndarray, torch.Tensor],
    network: nn.Module,
    ind_to_action: Callable = None,
) -> int:
    """
    Given a state and a neural network, choose the next action greedily
    by selecting the action with the highest probability.
    Args:
        state: Observed state of the environment
        network: Torch network used for selecting action.
        ind_to_action: function which maps from the neural network indexes
            to an action (in case the action indexing and values are different).
            By default, the neural network index will be returned.

    Returns:
        action to take
    """
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state)
    logits = network(state)
    action_ind = np.argmax(logits).item()
    action = action_ind
    if ind_to_action is not None:
        action = ind_to_action(action_ind)
    return action


def choose_action_sample(
    state: Union[np.ndarray, torch.Tensor],
    network: nn.Module,
    ind_to_action: Callable = None,
) -> int:
    """
    Given a state and a neural network, choose next action by sampling.

    We treat the `network` outputs as logits, and sample from the distribution
    according to those probabilities.
    Args:
        state: Observed state of the environment
        network: Torch network used for selecting action.
        ind_to_action: function which maps from the neural network indexes
            to an action (in case the action indexing and values are different).
            By default, the neural network index will be returned.

    Returns:
        action to take
    """
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state)
    action_dist = get_policy_discrete(state, network)
    action_ind = action_dist.sample().item()
    action = action_ind
    if ind_to_action is not None:
        action = ind_to_action(action_ind)
    return action


def get_policy_discrete(
    state: Union[np.ndarray, torch.Tensor], network: nn.Module
) -> torch.distributions.Distribution:
    """
    Get the policy for this network

    The 'policy' is a probability distribution over choices.
    For discrete options, this will be a Categorical distribution.

    Args:
        state: Observed environment state
        network: Neural network used for choosing action.
        The output of this network is treated as logits.

    Returns:
        A distribution over choices.
    """
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state)
    logits = network(state)
    action_dist = Categorical(logits=logits)
    return action_dist


class Trainer(ABC):
    """
    Base class for trainers.

    A trainer implements a training algorithm, such as DQN. The trainer
    has infrastructure for callbacks before/after epochs, as well as
    the whole training process.

    Attributes:
        env (gym.Env): Training environment. Is expected to behave like a
         `gym.env`, an `action_space` capable of sampling, and methods
         `reset` and `step` .
        network (nn.Module): The network being trained
        optimizer (torch.optim.Optimizer): The `optimizer` used for training.
        memory (Any): Default None. Here for subclasses that want to use a memory.
            The exact usage of that memory is left to subclasses.
        train_kwargs (Dict):
            The dictionary of keyword arguments passed to `trained_epoch`
        ind_to_action (Callable):
            Maps output of `network` to an action used in `env.step`.
            Default is to use the output directly.
        callbacks (Sequence[Callback]): S
            equence of `Callback`s used during training.


    Example:

    env = gym.make("CartPole-v1")
    # Note: This extremely simple network is unlikely to solve any task
    network = nn.Linear(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    callbacks = [PrintResults()]
    trainer = TrainerSubclass(env, network, optimizer, callbacks=callbacks)
    """

    def __init__(
        self,
        env: "gym.Env",
        network: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        ind_to_action: Callable = None,
        callbacks: Iterable["Callback"] = (),
    ):
        self.env = env
        self.network = network
        self.target_network = copy.deepcopy(network)
        self.optimizer = optimizer
        self.memory = None
        self.train_kwargs = {}
        self.ind_to_action = ind_to_action
        self.callbacks = []
        for callback in callbacks:
            self.add_callback(callback)

        # Here in case we want an EarlyStoppingCallback.
        self.stop_training = False

    def add_callback(self, callback: "Callback"):
        callback.set_trainer(self)
        self.callbacks.append(callback)

    def on_train_start(self):
        self.stop_training = False
        for callback in self.callbacks:
            callback.on_train_start()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_epoch_start(self, epoch_num: int):
        for callback in self.callbacks:
            callback.on_epoch_start(epoch_num)

    def on_epoch_end(self, epoch_num: int, epoch_results: Dict):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch_num, epoch_results)

    def train(self, epochs: int, **train_kwargs):
        self.train_kwargs = train_kwargs
        self.on_train_start()

        for epoch_num in tqdm(range(epochs)):

            self.on_epoch_start(epoch_num)

            epoch_results = self.train_epoch(**self.train_kwargs)

            self.on_epoch_end(epoch_num, epoch_results)

            if self.stop_training:
                break

        self.on_train_end()

        return self.network

    @abc.abstractmethod
    def train_epoch(self, *args, **kwargs):
        """Train for a single epoch.

        The definition of 'epoch' may vary by subclass.
        For the DoubleDQNTrainer it will be the set of events between
        when the environment is initialized and when it's done.
        For the SimplePolicyGradient, an epoch lasts until the memory buffer
        is full.
        """


class DoubleDQN(Trainer):
    """
    Class for implementing Double-DQN Training
    See https://www.nature.com/articles/nature14236

    Implementation uses experience replay, and chooses
    actions via epsilon-greedy method.
    """

    def on_train_start(self):
        """Initialize and/or clear memory buffer"""
        if self.memory is None:
            self.memory = DoubleDQN.ExperienceReplayMemory()
        self.memory.clear()
        return super().on_train_start()

    def on_epoch_end(self, epoch_num: int, epoch_results: Dict):
        """
        Update the target network every `update_target_interval` epochs.
        """

        # Could/should this be implemented with a callback?
        update_target_interval = self.train_kwargs.get("update_target_interval", 20)
        if epoch_num > 0 and epoch_num % update_target_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return super().on_epoch_end(epoch_num, epoch_results)

    def train_epoch(
        self,
        gamma: float = 1.0,
        batch_size: int = 150,
        epsilon: float = 0.0,
        max_steps: int = None,
        **kwargs,
    ):  # pylint: disable=unused-argument
        """
        Train for an epoch of our environment
        Args:
            gamma: Time discount parameter.
                value(t) = reward + gamma*value(t+1)
            batch_size: Batch size used for experience replay
            epsilon: epsilon used in epsilon-greedy.
                Default is 0, which is pure-greedy.
            max_steps: Maximum steps per epoch.
                Default is None, we let the epoch run until the environment says
                it's done.
            **kwargs: Other keyword arguments can be accepted without errors,
            though they will not be used.
        """

        state = self.env.reset()
        action_space = self.env.action_space
        b_idx = np.arange(batch_size)

        loss_obj = nn.MSELoss()

        done = False
        epoch_results = {"steps": 0, "total_reward": 0.0, "mean_loss": 0.0}
        step_iter = 0
        total_reward = total_loss = 0.0
        while not done:
            with torch.no_grad():
                if random.random() < epsilon:
                    action = action_space.sample()
                else:
                    action = choose_action_greedy(
                        state, self.network, self.ind_to_action
                    )

            succ_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if max_steps is not None:
                # Treat hitting the maximum as a regular end of episode
                done &= step_iter < max_steps

            self.memory.append(state, action, reward, succ_state, done)
            state = succ_state.copy()
            if len(self.memory) < batch_size:
                continue

            sample_tuple = self.memory.sample(batch_size)
            b_states, b_actions, b_rewards, b_succs, b_dones = sample_tuple

            # Update value function
            with torch.no_grad():
                network_actions = torch.argmax(self.network(b_succs), dim=1)
                all_target_values = self.target_network(b_succs)
                sel_target_values = all_target_values[b_idx, network_actions]

                final_mod = 1 - b_dones.to(int)
                tot_target_values = b_rewards + final_mod * gamma * sel_target_values

            # Calculate the predicted Q(state, action) values
            network_outs = self.network(b_states)
            predicted_value = network_outs[b_idx, b_actions]

            loss_tensor = loss_obj(predicted_value, tot_target_values)
            total_loss += loss_tensor.item()

            self.optimizer.zero_grad()
            loss_tensor.backward()
            self.optimizer.step()

            step_iter += 1

        epoch_results["steps"] = step_iter
        epoch_results["total_reward"] = total_reward
        if step_iter > 0:
            epoch_results["mean_loss"] = total_loss / step_iter

        return epoch_results

    class ExperienceReplayMemory:
        """
        Memory buffer used for experience replay

        The idea behind experience replay staring a large number of state transitions,
        and updating once those reach a certain size. These transition are *not*
        sequential, and may include many episodes.

        References:
            https://arxiv.org/abs/1707.01495v3
            https://arxiv.org/abs/1712.01275
        """

        def __init__(
            self, max_length: int = 100_000, float_type: torch.dtype = torch.float64
        ):
            """
            Args:
                max_length: Maximum size of the memory buffer
                float_type: torch.dtype to using for floating points
            """
            self.max_length = max_length
            self.deq = deque(maxlen=max_length)
            self.float_type = float_type

        def __len__(self):
            return len(self.deq)

        def clear(self):
            self.deq.clear()

        def append(
            self,
            state: Union[np.ndarray, torch.Tensor],
            action: int,
            reward: float,
            successor_state: Union[np.ndarray, torch.Tensor],
            is_terminal: bool,
        ):
            """
            Add a single transition to buffer

            Args:
                state: Observed state of the environment
                action: action we took at `state`
                reward: reward we received after taking `action`
                successor_state: state we transitioned into
                is_terminal: Whether `successor_state` was terminal, that is,
                the environment is now done.
            """

            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state)
            if isinstance(successor_state, np.ndarray):
                successor_state = torch.from_numpy(successor_state)

            to_add = (
                state.to(self.float_type),
                action,
                reward,
                successor_state.to(self.float_type),
                is_terminal,
            )
            self.deq.append(to_add)

        def sample(self, size: int) -> Tuple[torch.Tensor, ...]:
            """
            Randomly sample entries from the memory.
            Note that this will always include the most recently added.
            Args:
                size: Number to sample

            Returns:
                Tuple[states, actions, rewards, successive_states, is_terminals]
                All are torch.Tensor of various torch.dtype.
            """
            num_to_take = min(size, len(self.deq))

            # Use random.choices to sample with replacement
            choices = random.choices(self.deq, k=num_to_take)

            # Always include the latest
            # Per https://arxiv.org/abs/1712.01275
            choices[0] = self.deq[-1]

            return self._select(choices)

        def _select(self, choices: Sequence[Tuple]) -> Tuple:
            """Reformat choices so that instead of a sequence
            of individual transitions we have a set of 5 batched tensors.
            """
            tot_num_taken = len(choices)

            states = [torch.Tensor()] * tot_num_taken
            actions = [0] * tot_num_taken
            rewards = [0] * tot_num_taken
            succ_states = [torch.Tensor()] * tot_num_taken
            is_terms = [False] * tot_num_taken
            for ind, tup in enumerate(choices):
                states[ind] = tup[0]
                actions[ind] = tup[1]
                rewards[ind] = tup[2]
                succ_states[ind] = tup[3]
                is_terms[ind] = tup[4]

            states = torch.stack(states)
            succ_states = torch.stack(succ_states)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=self.float_type)
            is_terms = torch.tensor(is_terms, dtype=torch.bool)

            return states, actions, rewards, succ_states, is_terms


class SimplePolicyGradient(Trainer):
    """
    Class for implementing the simplest policy gradient training
    as defined in (retrieved Aug. 18, 2022):
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html,
    https://archive.ph/sGRbS
    """

    def on_train_start(self):
        if self.memory is None:
            self.memory = SimplePolicyGradient.EpisodicMemory()
        self.memory.clear()
        return super().on_train_start()

    def train_epoch(
        self,
        gamma: float = 1.0,
        batch_size: int = 150,
        epsilon: float = 0.0,
        max_steps_per_episode: int = None,
        **kwargs,
    ):  # pylint: disable=unused-argument
        """
        Train for an epoch.

        For `SimplePolicyGradient` this can be any number of environment episodes.
        We keep training until our memory has at least `batch_size` entries,
        then do an update step.

        Args:
            gamma: Time discount parameter.
                value(t) = reward + gamma*value(t+1)
            batch_size: Batch size used for episodic memory.
            epsilon: epsilon used in epsilon-greedy.
                Default is 0, which is pure-greedy.
            max_steps_per_episode: Maximum steps per episode.
                (episode, not epoch).
                Default is None, we continue training until the memory is full.
            **kwargs: Other keyword arguments can be accepted without errors,
            though they will not be used.
        """

        self.memory.clear()
        action_space = self.env.action_space

        epoch_results = {"steps": 0, "total_reward": 0.0, "mean_loss": 0.0}
        total_steps = 0
        total_reward = 0
        total_loss = 0

        # We loop through many episodes until our memory buffer is full,
        # and define those N-episodes as an epoch.
        epoch_done = False
        while not epoch_done:
            state = self.env.reset()
            episode_done = False
            episode_reward = 0.0
            step_iter = 0
            states = []
            actions = []
            while not episode_done:
                with torch.no_grad():
                    if random.random() < epsilon:
                        action = action_space.sample()
                    else:
                        action = choose_action_sample(
                            state, self.network, self.ind_to_action
                        )

                succ_state, reward, episode_done, _ = self.env.step(action)
                episode_reward += reward

                states.append(state)
                actions.append(action)
                step_iter += 1

                state = succ_state.copy()

                if max_steps_per_episode is not None:
                    # Treat hitting the maximum as a regular end of episode
                    episode_done &= step_iter < max_steps_per_episode

            self.memory.add_episode(states, actions, episode_reward)
            total_steps += step_iter
            total_reward += episode_reward

            # Stop once our memory buffer has enough
            epoch_done = len(self.memory) >= batch_size

        sample_tuple = self.memory.get_all()
        b_states, b_actions, b_weights = sample_tuple

        # Compute the loss
        logp = get_policy_discrete(b_states, self.network).log_prob(b_actions)
        loss_tensor = -1 * (logp * b_weights).mean()
        total_loss += loss_tensor.item()

        self.optimizer.zero_grad()
        loss_tensor.backward()
        self.optimizer.step()

        epoch_results["steps"] = total_steps
        epoch_results["total_reward"] = total_reward
        if total_steps > 0:
            epoch_results["mean_loss"] = total_loss / total_steps

        return epoch_results

    class EpisodicMemory:
        """
        Class used for storing results episode-by-episode
        """

        def __init__(self, float_type: torch.dtype = torch.float64):
            """
            Args:
                float_type: torch.dtype to using for floating points
            """
            self.float_type = float_type
            self.states = []
            self.actions = []
            self.weights = []

        def __len__(self):
            return len(self.states)

        def clear(self):
            """Clear memory"""
            self.states = []
            self.actions = []
            self.weights = []

        def add_episode(
            self,
            states: List[np.ndarray],
            actions: List[int],
            episode_reward: float,
        ) -> int:
            """
            Add results from a single episode to memory
            Args:
                states: Sequence of observed states of the environment.
                actions: Sequence of actions we took in the provided states.
                episode_reward: Total reward over the episode.

            Note:
                len(states) must equal len(actions).

            Returns:
                Number of states/actions added.
            """
            if len(states) != len(actions):
                err_str = f"State size {len(states)}"
                err_str += f" must be the same as action size {len(actions)}"
                raise ValueError(err_str)

            num_to_add = len(states)
            new_weights = [episode_reward] * num_to_add

            self.states += states
            self.actions += actions
            self.weights += new_weights

            return num_to_add

        def get_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Get entire memory, formatted as Torch tensors.

            The weight of each transition is the total reward over the episode
            from which that transition took place.

            Returns:
                Tuple[states, actions, weights]
            """
            states_t = torch.as_tensor(np.array(self.states), dtype=self.float_type)
            actions_t = torch.as_tensor(self.actions, dtype=torch.int32)
            weights_t = torch.as_tensor(self.weights, dtype=self.float_type)
            return states_t, actions_t, weights_t
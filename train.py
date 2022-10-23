__doc__ = """ Script used for training models.

Intended to demonstrate usage of `trainers` and `callbacks`.
"""

import datetime
import os
import time
from typing import Iterable, Type

import gym
import numpy as np
import torch
from ray import tune
from torch import nn

from callbacks import DecayParameter, TensorboardCallback
from parameters import get_parameters
from trainers import DoubleDQN, SimplePolicyGradient

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda:0"


def build_network(
    input_size: int,
    output_size: int,
    hidden_layers: Iterable = (64, 128),
    activation: nn.Module = nn.ReLU,
) -> nn.Module:
    """
    Create a dense neural network with hidden layers that are all the same size

    Example:
        my_network = build_network(3, 4, hidden_layers=2,
                                  hdim=64, activation=nn.ReLU6)
        This would create a 4-layer network:
        (input, hidden_0, hidden_1, output) with ReLU6 activations.

    Args:
        input_size: Size of input
        output_size: Size of output
        hidden_layers: Iterable of dimensions of hidden layers.
            Must be at least length 1.
        activation: Activation function. Must be a callable, such as nn.ReLU.
            Used after all layers except the output.

    Returns:

    """

    if len(hidden_layers) < 1:
        raise ValueError("Must provide at least 1 hidden layer")

    all_layers = []
    internal_layer_sizes = [input_size] + list(hidden_layers)
    num_internal_layers = len(internal_layer_sizes)

    for ind in range(num_internal_layers - 1):
        dim1 = internal_layer_sizes[ind]
        dim2 = internal_layer_sizes[ind + 1]
        all_layers.extend([nn.Linear(dim1, dim2), activation()])

    all_layers.append(nn.Linear(internal_layer_sizes[-1], output_size))

    seq_network = nn.Sequential(*all_layers)
    network = seq_network.double().to(DEVICE)
    return network


def train_example(
    trainer_class: Type,
    env_name: str,
    train_kwargs,
    decay_parameters=(),
    log_dir=None,
    checkpoint_dir=None,
    eval_interval=None,
):
    """Example of training a network using these classes. We use the OpenAI gym."""
    env = gym.make(env_name)

    epochs = train_kwargs.pop("epochs")
    lr = train_kwargs.pop("lr")

    network = build_network(env.observation_space.shape[0], env.action_space.n)
    # print(f"Network: ")
    # print(network)

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        network.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # current_time = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    callbacks = []
    if log_dir is not None:
        tboard_callback = TensorboardCallback(log_dir)
        callbacks = [tboard_callback]

    if decay_parameters:
        for dp in decay_parameters:
            callbacks.append(DecayParameter(**dp))

    # Train my network
    trainer = trainer_class(
        env,
        network,
        optimizer,
        callbacks=callbacks,
        loss=train_kwargs.pop("loss", None),
    )
    # trainer = DoubleDQNTrainer(env, network, optimizer, callbacks=callbacks)
    # trainer = SimplePolicyGradient(env, network, optimizer, callbacks=callbacks)

    trainer.train(epochs, eval_interval=eval_interval, **train_kwargs)
    network.eval()
    # Training finished

    if checkpoint_dir is not None:
        with tune.checkpoint_dir(step=epochs) as save_checkpoint_dir:
            # print(save_checkpoint_dir)
            path = os.path.join(save_checkpoint_dir, "checkpoint")
            torch.save((network.state_dict(), optimizer.state_dict()), path)

    return network


def run_env(env_name, network, temperature=0.0, verbose=False):
    """Run the network in the environment once"""
    env = gym.make(env_name)
    state = env.reset()
    done = False

    if verbose:
        np.set_printoptions(precision=4)
        env.render()

    steps = total_reward = 0
    temp_decay = 1.0
    while not done:
        q_values = network(torch.tensor(state))
        if temperature == 0:
            action = torch.argmax(q_values).item()
        else:
            action = (
                torch.distributions.Categorical(logits=q_values / temperature)
                .sample()
                .item()
            )
        state, reward, done, _ = env.step(action)

        steps += 1
        total_reward += reward
        temperature *= temp_decay

        if verbose:
            env.render()
            time.sleep(0.05)
            if steps % 5 == 0:
                c_str = f"Step {steps}"
                c_str += f", q_value {q_values.detach().numpy()}"
                c_str += f", action {action}"
                c_str += f", current reward {reward}"
                c_str += f", total reward {total_reward}"
                print(c_str)

    return steps, total_reward


if __name__ == "__main__":

    # Classic control environments with discrete action spaces:
    # "Acrobot-v1", "CartPole-v1", "MountainCar-v0"
    ENV_NAME = "MountainCar-v0"
    # ind_to_action = lambda x: x - 1
    # ENV_NAME = "CartPole-v1"
    # ind_to_action = None

    # TrainerClass = DoubleDQN
    TrainerClass = SimplePolicyGradient
    trainer_name = TrainerClass.__name__

    # tag = "rtg"
    tag = "debug"

    train_kwargs, decay_parameters = get_parameters(trainer_name, ENV_NAME)

    exp_name = f"{trainer_name}_{ENV_NAME}_{tag}"
    network_path = f"models/{exp_name}.pt"
    log_dir = os.path.join("logs", exp_name)
    DO_TRAIN = False or not os.path.exists(network_path)
    DO_RUN = True
    if DO_TRAIN:
        main_network = train_example(
            TrainerClass,
            ENV_NAME,
            train_kwargs,
            decay_parameters=decay_parameters,
            log_dir=log_dir,
            eval_interval=10,
        )
        print(f"Saving network to {network_path}")
        torch.save(main_network, network_path)

    if DO_RUN:
        print(f"Loading network from {network_path}")
        main_network = torch.load(network_path)
        n_steps, outer_total_reward = run_env(ENV_NAME, main_network,
                                              temperature=train_kwargs.get("temperature", 0),
                                              verbose=True)
        p_str = f"Finished {exp_name} after {n_steps} steps."
        p_str += f"  Total reward {outer_total_reward}"
        print(p_str)
    print(f"{datetime.datetime.now()} Finished")

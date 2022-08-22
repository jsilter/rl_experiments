__doc__ = """ Script used for training models.

Intended to demonstrate usage of `trainers` and `callbacks`.
"""

import os
import time
from typing import Type

import gym
import torch
from ray import tune
from torch import nn

from callbacks import DecayParameter, TensorboardCallback
from trainers import DoubleDQN, SimplePolicyGradient

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda:0"


def build_network(
    input_size: int,
    output_size: int,
    hidden_layers=2,
    hdim=64,
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
        hidden_layers: Number of hidden layers. Note that 0 is allowed,
            and creates a two-layer (input and output) network.
        hdim: Dimension of hidden layers.
        activation: Activation function. Must be a callable, such as nn.ReLU.
            Used after all layers except the output.

    Returns:

    """

    if hidden_layers < 0:
        raise ValueError(f"hidden_layers must be non-negative, not {hidden_layers}")

    all_layers = [
        nn.Linear(input_size, hdim),
        activation(),
    ]

    for _ in range(hidden_layers):
        all_layers.extend([nn.Linear(hdim, hdim), activation()])

    all_layers.append(nn.Linear(hdim, output_size))

    seq_network = nn.Sequential(*all_layers)
    network = seq_network.double().to(DEVICE)
    return network


def train_example(
    trainer_class: Type,
    env_name: str,
    train_kwargs,
    eps_dict=None,
    log_dir=None,
    checkpoint_dir=None,
):
    """Example of training a network using these classes. We use the OpenAI gym."""
    env = gym.make(env_name)

    epochs = train_kwargs.pop("epochs")
    lr = train_kwargs.pop("lr")

    network = build_network(env.observation_space.shape[0], env.action_space.n)

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

    if eps_dict:
        callbacks.append(DecayParameter(**eps_dict))

    # Train my network
    trainer = trainer_class(env, network, optimizer, callbacks=callbacks)
    # trainer = DoubleDQNTrainer(env, network, optimizer, callbacks=callbacks)
    # trainer = SimplePolicyGradient(env, network, optimizer, callbacks=callbacks)

    trainer.train(epochs, **train_kwargs)
    network.eval()
    # Training finished

    if checkpoint_dir is not None:
        with tune.checkpoint_dir(step=epochs) as save_checkpoint_dir:
            # print(save_checkpoint_dir)
            path = os.path.join(save_checkpoint_dir, "checkpoint")
            torch.save((network.state_dict(), optimizer.state_dict()), path)

    return network


def run_env(env_name, network, verbose=False):
    """Run the network in the environment once"""
    env = gym.make(env_name)
    state = env.reset()
    done = False

    if verbose:
        env.render()

    c_steps = c_total_reward = 0
    while not done:
        q_values = network(torch.tensor(state))
        action = torch.argmax(q_values).item()
        state, reward, done, _ = env.step(action)

        c_steps += 1
        c_total_reward += reward

        if verbose:
            env.render()
            time.sleep(0.05)
            if c_steps % 5 == 0:
                c_str = f"Step {c_steps}"
                c_str += f", current reward {reward}"
                c_str += f", total reward {c_total_reward}"
                print(c_str)

    return c_steps, c_total_reward


if __name__ == "__main__":
    # Classic control environments with discrete action spaces:
    # "Acrobot-v1", "CartPole-v1", "MountainCar-v0"
    ENV_NAME = "CartPole-v1"

    # TrainerClass = DoubleDQN
    TrainerClass = SimplePolicyGradient
    trainer_name = TrainerClass.__name__

    if trainer_name == "SimplePolicyGradient":
        # These work well for SimplePolicyGradient cartpole
        cp_train_kwargs = {
            "epochs": 4000,
            "lr": 1e-4,
        }
        cp_eps_dict = {
            "name": "epsilon",
            "init": 0.50,
            "decay": 0.98,
            "min_value": 0.01,
        }

        use_train_kwargs = cp_train_kwargs
        use_eps_dict = cp_eps_dict
    elif trainer_name == "DoubleDQN":
        # These work well for DoubleDQN cartpole
        cp_train_kwargs = {
            "epochs": 250,
            "lr": 1e-4,
            "gamma": 0.99,
        }
        cp_eps_dict = {
            "name": "epsilon",
            "init": 0.50,
            "decay": 0.98,
            "min_value": 0.01,
        }

        use_train_kwargs = cp_train_kwargs
        use_eps_dict = cp_eps_dict

    if ENV_NAME == "Acrobot-v1":
        pass
    elif ENV_NAME == "MountainCar-v0":
        pass

    network_path = f"{trainer_name}_{ENV_NAME.lower()}_latest.pt"
    log_dir = os.path.join("logs", trainer_name, ENV_NAME)
    DO_TRAIN = True or not os.path.exists(network_path)
    DO_RUN = True
    if DO_TRAIN:
        main_network = train_example(
            TrainerClass,
            ENV_NAME,
            use_train_kwargs,
            eps_dict=use_eps_dict,
            log_dir=log_dir,
        )
        torch.save(main_network, network_path)

    if DO_RUN:
        main_network = torch.load(network_path)
        n_steps, outer_total_reward = run_env(ENV_NAME, main_network, verbose=True)
        p_str = f"Finished {trainer_name} - {ENV_NAME} after {n_steps} steps."
        p_str += f"  Total reward {outer_total_reward}"
        print(p_str)

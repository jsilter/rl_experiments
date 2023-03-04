__doc__ = """ Store hyper-parameters for models / tasks.
"""

from typing import Dict, List, Tuple

from torch import nn


def get_parameters(trainer_name: str, env_name: str) -> Tuple[Dict, List]:
    """
    Retrieve appropriate hyper-parameters for the given trainer and task.
    Args:
        trainer_name: Name of trainer to use (e.g. DoubleDQN)
        env_name: Name of environment in which to train (e.g. CartPole-v1)

    Returns:
        train_kwargs : training keyword arguments
        Decay parameters : List of parameters which decay over the course of training.
    """
    param_key = (trainer_name, env_name)

    if param_key == ("DoubleDQN", "CartPole-v1"):
        train_kwargs = {
            "epochs": 250,
            "lr": 1e-4,
            "gamma": 0.99,
        }
        eps_dict = {
            "name": "epsilon",
            "init": 0.50,
            "decay": 0.98,
            "min_value": 0.01,
        }
        decay_parameters = [eps_dict]
    elif param_key == ("SimplePolicyGradient", "CartPole-v1"):
        train_kwargs = {
            "epochs": 4000,
            "lr": 1e-4,
        }
        eps_dict = {
            "name": "epsilon",
            "init": 0.50,
            "decay": 0.98,
            "min_value": 0.01,
        }
        decay_parameters = [eps_dict]
    elif param_key == ("DoubleDQN", "Acrobot-v1"):
        train_kwargs = {
            "epochs": 1000,
            "lr": 1e-4,
            "gamma": 0.99,
            "loss": nn.SmoothL1Loss(),
        }

        eps_dict = {
            "name": "epsilon",
            "init": 0.00,
            "decay": 0.988,
            "min_value": 0.00,
        }

        temp_dict = {
            "name": "temperature",
            "init": 50.0,
            "decay": 0.988,
            "min_value": 0.0,
        }
        decay_parameters = [eps_dict, temp_dict]
    elif param_key == ("SimplePolicyGradient", "Acrobot-v1"):
        train_kwargs = {"epochs": 4000, "lr": 8e-5, "temperature": 1.0}
        eps_dict = {
            "name": "epsilon",
            "init": 0.50,
            "decay": 0.98,
            "min_value": 0.01,
        }
        decay_parameters = [eps_dict]
    elif param_key == ("DoubleDQN", "MountainCar-v0"):
        train_kwargs = {
            "epochs": 1000,
            "lr": 5e-4,
            "gamma": 0.99,
            "loss": nn.SmoothL1Loss(),
        }
        eps_dict = {
            "name": "epsilon",
            "init": 0.1,
            "decay": 0.988,
            "min_value": 0.00,
        }
        temp_dict = {
            "name": "temperature",
            "init": 50.0,
            "decay": 0.988,
            "min_value": 0.0,
        }
        decay_parameters = [eps_dict, temp_dict]
    elif param_key == ("SimplePolicyGradient", "MountainCar-v0"):
        train_kwargs = {"epochs": 4000, "lr": 5e-5}
        eps_dict = {
            "name": "epsilon",
            "init": 0.1,
            "decay": 0.99,
            "min_value": 0.00,
        }
        temp_dict = {
            "name": "temperature",
            "init": 50.0,
            "decay": 0.99,
            "min_value": 1.0,
        }
        decay_parameters = [eps_dict, temp_dict]
    else:
        raise ValueError(f"No parameters set for {param_key}")

    return train_kwargs, decay_parameters

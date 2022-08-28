__doc__ = """ Hyper-parameter tuning"""

import torch
from ray import tune

from train import run_env, train_example
from trainers import DoubleDQN, SimplePolicyGradient

if __name__ == "__main__":
    # Use Ray  for hyperparameter tuning
    # https://docs.ray.io/en/latest/index.html
    # https://docs.ray.io/en/latest/tune/api_docs/search_space.html
    TrainerClass = DoubleDQN
    epochs = 800
    # TrainerClass = SimplePolicyGradient
    # epochs = 4000

    search_space = {
        # "lr": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
        "epsilon": tune.grid_search([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    }
    tag = "eps_ep800"

    trainer_name = TrainerClass.__name__
    ENV_NAME = "Acrobot-v1"
    exp_name = f"{trainer_name}_{ENV_NAME}_{tag}"
    local_checkpoint_dir = "checkpoints"
    num_samples = 3

    # These get over-written by new parameters from the search space, if applicable
    train_kwargs = {
        "epochs": epochs,
        "lr": 1e-4,
        "gamma": 0.99,
        "loss": torch.nn.SmoothL1Loss(),
    }

    # These may get overwritten
    eps_dict = {"name": "epsilon", "init": 0.00, "decay": 0.988, "min_value": 0.00}

    def objective(config, checkpoint_dir=None):
        cur_train_kwargs = train_kwargs.copy()
        cur_train_kwargs.update(config)

        temp_dict = {
            "name": "temperature",
            "init": 5.0,
            "decay": 0.988,
            "min_value": 0.0,
        }
        cur_eps_dict = eps_dict.copy()
        cur_eps_dict["init"] = config["epsilon"]

        decay_parameters = [cur_eps_dict, temp_dict]

        # Note: I could save tensorboard logs with:
        # log_dir = os.path.join(checkpoint_dir, "logs")
        network = train_example(
            TrainerClass,
            ENV_NAME,
            cur_train_kwargs,
            decay_parameters=decay_parameters,
            checkpoint_dir=checkpoint_dir,
        )
        steps, reward = run_env(ENV_NAME, network, verbose=False)
        tune.report(steps=steps, reward=reward)

    resources_pt = {"cpu": 1, "memory": 3e9}
    analysis = tune.run(
        objective,
        metric="reward",
        mode="max",
        name=exp_name,
        resources_per_trial=resources_pt,
        config=search_space,
        local_dir=local_checkpoint_dir,
        num_samples=num_samples,
        resume="AUTO",
        raise_on_failed_trial=False,
    )

    print("Best config is:", analysis.best_config)
    output_path = f"{exp_name}_tuning_results.tsv"
    print(f"Saving results to {output_path}")
    analysis.results_df.to_csv(output_path, sep="\t")

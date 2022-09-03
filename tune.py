__doc__ = """ Hyper-parameter tuning"""

import os

import pandas as pd
import torch
from ray import tune

from parameters import get_parameters
from train import run_env, train_example
from trainers import DoubleDQN, SimplePolicyGradient

if __name__ == "__main__":
    # Use Ray  for hyperparameter tuning
    # https://docs.ray.io/en/latest/index.html
    # https://docs.ray.io/en/latest/tune/api_docs/search_space.html
    # TrainerClass = DoubleDQN
    # epochs = 800
    TrainerClass = SimplePolicyGradient
    epochs = None

    search_space = {
        "lr": tune.grid_search([1e-5, 5e-5, 8e-5, 1e-4, 3e-4, 5e-4, 1e-3, 5e-3]),
        # "epsilon": tune.grid_search([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    }
    tag = "lr_ep4k"

    trainer_name = TrainerClass.__name__
    ENV_NAME = "Acrobot-v1"
    exp_name = f"{trainer_name}_{ENV_NAME}_{tag}"
    output_dir = "tuning_results"
    output_stub = f"{output_dir}/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    local_checkpoint_dir = "checkpoints"
    num_samples = 5

    def objective(config, checkpoint_dir=None):
        train_kwargs, decay_parameters = get_parameters(trainer_name, ENV_NAME)
        cur_train_kwargs = train_kwargs.copy()
        cur_train_kwargs.update(config)
        if epochs:
            cur_train_kwargs.update({"epochs": epochs})

        for decay_parm in decay_parameters:
            cur_name = decay_parm["name"]
            if cur_name in config:
                decay_parm["init"] = config[cur_name]

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
    output_path = f"{output_stub}_tuning_results.tsv"
    print(f"Saving results to {output_path}")
    analysis.results_df.to_csv(output_path, sep="\t")

    group_cols = [f"config.{x}" for x in sorted(search_space.keys())]
    results_df = analysis.results_df
    mean_performance = results_df.groupby(group_cols)["reward"].mean()
    std_performance = results_df.groupby(group_cols)["reward"].std()
    counts = results_df.groupby(group_cols)["reward"].count()

    collated_df = pd.concat([mean_performance, std_performance, counts], axis=1)
    collated_df.columns = ["mean", "std", "count"]
    collated_df.sort_values("mean", ascending=False, inplace=True)
    collated_df.to_csv(f"{output_stub}_collated_results.tsv", sep="\t")

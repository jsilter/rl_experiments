__doc__ = """ Hyper-parameter tuning"""

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
        "lr": tune.grid_search([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
    }

    trainer_name = TrainerClass.__name__
    ENV_NAME = "CartPole-v1"
    exp_name = f"{trainer_name}_{ENV_NAME}"
    local_checkpoint_dir = "checkpoints"
    num_samples = 30

    # These get over-written by new parameters from the search space, if applicable
    train_kwargs = {
        "epochs": epochs,
        "lr": 1e-4,
        "gamma": 0.7,
    }

    # These *do* matter, they are the same every time
    eps_dict = {"name": "epsilon", "init": 0.50, "decay": 0.98, "min_value": 0.01}

    def objective(config, checkpoint_dir=None):
        cur_train_kwargs = train_kwargs.copy()
        cur_train_kwargs.update(config)

        cur_eps_dict = eps_dict.copy()

        network = train_example(
            TrainerClass,
            ENV_NAME,
            cur_train_kwargs,
            eps_dict=cur_eps_dict,
            checkpoint_dir=checkpoint_dir,
        )
        steps, reward = run_env(ENV_NAME, network, verbose=False)
        tune.report(steps=steps, reward=reward)

    analysis = tune.run(
        objective,
        metric="reward",
        mode="max",
        name=exp_name,
        resources_per_trial={"cpu": 1},
        config=search_space,
        local_dir=local_checkpoint_dir,
        num_samples=num_samples,
        resume="AUTO",
    )

    print("Best config is:", analysis.best_config)
    analysis.results_df.to_csv(
        f"{trainer_name}_{ENV_NAME}_tuning_results.tsv", sep="\t"
    )

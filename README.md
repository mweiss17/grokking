# Grokking

## Installation

## Running locally

## Running on the cluster

## Running hyper-parameter searches


First we should construct a command that uses the `{job_idx}` to create different output experiment directories and sets the seed as follows. here we are sampling a learning rate from a log uniform distribution, and evaluating the `{lr}` with these different samples. When you run the following command, it should spit out 10 launch commands. Additionally, we are setting the group name and run name on wandb so that we can do hyper parameter sweeps and then not have an issue with mapping the config to the actual wandb runs on there.

`salvo -dry --- scripts/01-train.py ~/scratch/grokking-experiments/exp-{job_idx} --inherit templates/arithmetic --config.lr {lr} --config.seed {job_idx} --config._wandb_run_name exp-{job_idx} --config._wandb_group_name lr-sweep --config.seed {job_idx} --- generators/random_search.py -d lr~log_uniform[0.01,0.001] -n 10`

Then remove the `-dry` arg to actually submit these to slurm.

# Grokking

## Installation
clone the repo, cd into it, then `pip install -e .`. 

## Running locally
`python3 scripts/train.py experiments/exp-1 --inherit templates/arithmetic`


## Running on the cluster
You can submit your jobs to slurm like this:
`submit scripts/train.py ~/scratch/grokking-experiments/exp-1 --inherit templates/arithmetic`.
If you want to modify the type of node that will run your job, you can specify that with the commandline,
`submit scripts/train.py ~/scratch/grokking-experiments/exp-1 --inherit templates/arithmetic --mem_gb 8 --slurm_gres gpu:rtx8000:1 --cpus_per_task 12`.
By default, we use 8gb of memory and 12 cpus.

## Running hyper-parameter searches


First, we should construct a command that uses the `{job_idx}` to create different output experiment directories and sets the seed as follows. here we are sampling a learning rate from a log uniform distribution, and evaluating the `{lr}` with these different samples. When you run the following command, it should spit out 10 launch commands. Additionally, we are setting the group name and run name on wandb so that we can do hyper parameter sweeps and then not have an issue with mapping the config to the actual wandb runs on there.

`salvo -dry --- scripts/01-train.py ~/scratch/grokking-experiments/exp-{job_idx} --inherit templates/arithmetic --config.lr {lr} --config.seed {job_idx} --config._wandb_run_name exp-{job_idx} --config._wandb_group_name lr-sweep --config.seed {job_idx} --- generators/random_search.py -d lr~log_uniform[0.01,0.001] -n 10`

Remove the `-dry` arg to actually submit these to slurm. They will write logs out to the exp directory (e.g.`~/scratch/grokking-experiments/exp-1`), so you can see if something crashed. You can also check your running job with `squeue -u $USER`.

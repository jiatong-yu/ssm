#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=mteb_clustering
#SBATCH --output=slurm_outputs/output-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=36G
#SBATCH --time=0-10:00:00    # Run for 4 hours
#SBATCH --gres=gpu:1

# on average one GPU takes 5 cpu and 20GB memory

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=ALL
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=jiatongy@princeton.edu

conda activate .venv
python mteb_eval.py &


# Submit jobs.
wait; #Make sure to wait till all the runs have completed.

# Finish the script
exit 0
# SLURM submission scripts (reference only)

These are the exact SLURM batch scripts used to run PULSE on the authors' HPC
cluster. They are included for transparency about how the paper experiments were
launched, **not** as portable scripts.

Before running any of them on your own system you must edit:

- every absolute path (working directory, dataset roots, log locations),
- the Python interpreter / Conda environment line,
- the `#SBATCH` directives (`--partition`, `--gres`, `--time`, account) to match
  your scheduler and available GPUs.

For normal use you do not need SLURM at all. Run the commands in the main
[README](../../README.md) directly (training, inference, and evaluation all work
from the command line on a single machine).

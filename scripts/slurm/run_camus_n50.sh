#!/bin/bash
#SBATCH --job-name=camus_N50
#SBATCH --output=/SLURM/home/slurm_g202518690/student251/CVPR/logs/camus_N50_%j.out
#SBATCH --error=/SLURM/home/slurm_g202518690/student251/CVPR/logs/camus_N50_%j.err
#SBATCH --partition=interactive
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=01:45:00
PY=/SLURM/home/slurm_g202518690/student251/conda_envs/dinov3/bin/python3
export TRANSFORMERS_OFFLINE=1
export CAMUS_ROOT=/SLURM/home/slurm_g202518690/student251/CAMUS/database_nifti
cd /SLURM/home/slurm_g202518690/student251/CVPR
mkdir -p logs
$PY -c "
import eval_camus_fewshot as ev
ev.N_SHOTS = [50]
import sys
sys.argv = ['eval_camus_fewshot.py', '--ckpt_dir', 'folds', '--view', '2CH', '--epochs', '20']
ev.main()
"
echo "CAMUS N=50 DONE"

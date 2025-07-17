#!/bin/bash
#SBATCH --job-name=glv_simulation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_out/%A/%A_%a.out
#SBATCH --error=slurm_out/%A/%A_%a.err

source activate lnc
cd ../..

# First argument is the job list file
job_list_file="$1"
shift  # Remove job list file from the arg list

# Remaining args are forwarded to the Python script
mapfile -t jobs < "$job_list_file"

# Extract the interaction + chunk for this SLURM task
job="${jobs[$SLURM_ARRAY_TASK_ID]}"
interaction_suffix="${job%%:*}"  # Extract part before colon
chunk_val="${job##*:}"          # Extract part after colon

# Run the Python script with additional args
python ./synth/simulate/generate_data_GLV_fromInput.py \
  --chunk_num "$chunk_val" \
  --interactions "$interaction_suffix" \
  "$@"

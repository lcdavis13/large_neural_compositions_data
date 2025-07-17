#!/bin/bash
#SBATCH --job-name=glv_simulation
#SBATCH --array=0-21 # 0-29   # Specify the number of parallel jobs  # WARNING: Currently, if you use less than all Assemblage files, it will choose them alphabetically not ordinally 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_out/%A/%A_%a.out
#SBATCH --error=slurm_out/%A/%A_%a.err

source activate lnc
cd ../..
python ./synth/simulate/generate_data_GLV_fromInput.py --chunk_num $SLURM_ARRAY_TASK_ID --num_otus 256 --interactions "random-0" --assemblage_types "x0" --samples 5000 --time_file "t.csv"

# --replicator_normalization


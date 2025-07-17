#!/bin/bash

# Define interaction prefix and structure
interaction_prefix="random-"

# Declare suffixes and their chunks
declare -A chunk_specs
chunk_specs["0"]="1 test"
chunk_specs["1"]="1-20 test"
chunk_specs["2"]="1-20 test"
chunk_specs["3"]="1-20 test"

# Extra arguments to forward to Python
other_args=(
  --num_otus 256
  --samples 5000
  --assemblage_types "x0"
  --time_file "t.csv"
)

# Build job list
jobs=()
for suffix in "${!chunk_specs[@]}"; do
    chunks="${chunk_specs[$suffix]}"
    for entry in $chunks; do
        if [[ $entry =~ ^[0-9]+-[0-9]+$ ]]; then
            IFS="-" read -r start end <<< "$entry"
            for ((i=start; i<=end; i++)); do
                jobs+=("${interaction_prefix}${suffix}:${i}")
            done
        else
            jobs+=("${interaction_prefix}${suffix}:${entry}")
        fi
    done
done

# Save job list to file
job_list_file="job_list.txt"
printf "%s\n" "${jobs[@]}" > "$job_list_file"

# Submit job and forward extra args
num_jobs=${#jobs[@]}
sbatch --array=0-$((num_jobs - 1)) generate_datasets_helper.sh "$job_list_file" "${other_args[@]}"

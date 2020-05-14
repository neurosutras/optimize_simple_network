#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=simulate_simple_network_replay_"$2"_"$DATE"
export CONFIG_FILE_PATH="$1"
export LABEL="$2"
export PARAM_FILE_PATH="$3"
export MODEL_KEY="$4"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/src/optimize_simple_network/logs/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/src/optimize_simple_network/logs/$JOB_NAME.%j.e
#SBATCH -p development
#SBATCH -N 40
#SBATCH -n 2240
#SBATCH -t 2:00:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $SCRATCH/src/optimize_simple_network

ibrun -n 2240 python3 simulate_simple_network_replay.py --config-file-path=$CONFIG_FILE_PATH --verbose=1 --procs_per_worker=112 --export --num_input_patterns=1000 \
 --param_file_path=$PARAM_FILE_PATH --model_key=$MODEL_KEY
EOT

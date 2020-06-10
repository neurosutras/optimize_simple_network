#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"_"$3"
export JOB_NAME=simulate_simple_network_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
export NETWORK_ID="$3"
export PARAM_FILE_PATH="$4"
export MODEL_KEY="$5"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/src/optimize_simple_network/logs/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/src/optimize_simple_network/logs/$JOB_NAME.%j.e
#SBATCH -p development
#SBATCH -N 10
#SBATCH -n 560
#SBATCH -t 2:00:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $SCRATCH/src/optimize_simple_network

ibrun -n 560 python3 simulate_simple_network.py --config-file-path=$CONFIG_FILE_PATH --verbose=1 \
    --procs_per_worker=112 --export --num_trials=5 --param_file_path=$PARAM_FILE_PATH --model_key=$MODEL_KEY \
    --network_id=$NETWORK_ID --label=$NETWORK_ID --merge-output-files
EOT

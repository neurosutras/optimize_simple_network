#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL=simple_network_replay_"$2"_"$3"_"$4"
export JOB_NAME=simulate_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
export NETWORK_ID="$3"
export NETWORK_INSTANCE="$4"
export PARAM_FILE_PATH="$5"
export MODEL_KEY="$6"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/optimize_simple_network/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/optimize_simple_network/$JOB_NAME.%j.e
#SBATCH -p normal
#SBATCH -N 20
#SBATCH -n 1120
#SBATCH -t 1:00:00
#SBATCH --mail-user=neurosutras@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK2/optimize_simple_network

ibrun -n 1120 python3 simulate_simple_network_replay.py --config-file-path=$CONFIG_FILE_PATH --verbose=1 \
    --procs_per_worker=112 --export --num_trials=1000 --param_file_path=$PARAM_FILE_PATH \
    --model_key=$MODEL_KEY --network_id=$NETWORK_ID --network_instance=$NETWORK_INSTANCE --label=$LABEL \
    --merge-output-files --output-dir=$SCRATCH/data/optimize_simple_network
EOT

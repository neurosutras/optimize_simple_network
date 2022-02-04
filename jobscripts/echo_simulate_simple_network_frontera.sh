#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export CONFIG_FILE_PATH="$1"
export MODEL_LABEL="$2"
export NETWORK_ID="$3"
export NETWORK_INSTANCE_START="$4"
export PARAM_FILE_PATH="$5"
export MODEL_KEY="$6"
export LABEL=simple_network_"$MODEL_LABEL"_"$NETWORK_ID"
export JOB_NAME=simulate_"$LABEL"_"$DATE"

let "NETWORK_INSTANCE_END = $NETWORK_INSTANCE_START + 5"
export NETWORK_INSTANCE_END

sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/optimize_simple_network/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/optimize_simple_network/$JOB_NAME.%j.e
#SBATCH -p normal
#SBATCH -N 10
#SBATCH -n 560
#SBATCH -t 1:00:00
#SBATCH --mail-user=neurosutras@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK2/optimize_simple_network

for ((NETWORK_INSTANCE=$NETWORK_INSTANCE_START;NETWORK_INSTANCE<$NETWORK_INSTANCE_END;NETWORK_INSTANCE++))
do
  export NETWORK_INSTANCE
  echo ibrun -n 560 python3 simulate_simple_network.py --config-file-path=$CONFIG_FILE_PATH --verbose=1 \
      --procs_per_worker=112 --export --num_trials=5 --param_file_path=$PARAM_FILE_PATH --model_key=$MODEL_KEY \
      --network_id=$NETWORK_ID --network_instance=$NETWORK_INSTANCE --label="$LABEL"_"$NETWORK_INSTANCE" \
      --merge-output-files --output-dir=$SCRATCH/data/optimize_simple_network
done
EOT

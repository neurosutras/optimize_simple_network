#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL=simple_network_replay_"$3"
export JOB_NAME=decode_"$LABEL"_"$DATE"
export RUN_DATA_FILE_PATH="$1"
export REPLAY_DATA_FILE_PATH="$2"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/optimize_simple_network/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/optimize_simple_network/$JOB_NAME.%j.e
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -n 112
#SBATCH -t 1:00:00
#SBATCH --mail-user=neurosutras@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK/optimize_simple_network

ibrun -n 112 python3 decode_simple_network_replay.py --run-data-file-path=$RUN_DATA_FILE_PATH \
    --replay-data-file-path=$REPLAY_DATA_FILE_PATH --output-dir=$SCRATCH/data/optimize_simple_network --export
EOT

#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL=simple_network_replay_"$2"
export JOB_NAME=analyze_"$LABEL"_rhythmicity_"$DATE"
export REPLAY_DATA_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/optimize_simple_network/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/optimize_simple_network/$JOB_NAME.%j.e
#SBATCH -p normal
#SBATCH -N 3
#SBATCH -n 168
#SBATCH -t 0:30:00
#SBATCH --mail-user=neurosutras@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK2/optimize_simple_network

ibrun -n 168 python3 analyze_simple_network_replay_rhythmicity.py --replay-data-file-path=$REPLAY_DATA_FILE_PATH \
  --output-dir=$SCRATCH/data/optimize_simple_network --export
EOT

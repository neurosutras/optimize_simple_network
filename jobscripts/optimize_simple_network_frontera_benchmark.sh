#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_simple_network_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/optimize_simple_network/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/optimize_simple_network/$JOB_NAME.%j.e
#SBATCH -p development
#SBATCH -N 40
#SBATCH -n 2240
#SBATCH -t 0:30:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK/optimize_simple_network

ibrun -n 2240 python3 -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=$SCRATCH/data/optimize_simple_network --pop_size=200 --max_iter=1 --path_length=1 --disp \
  --procs_per_worker=112
EOT

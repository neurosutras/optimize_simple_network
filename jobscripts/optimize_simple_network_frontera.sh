#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_simple_network_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/src/optimize_simple_network/logs/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/src/optimize_simple_network/logs/$JOB_NAME.%j.e
#SBATCH -p normal
#SBATCH -N 40
#SBATCH -n 2240
#SBATCH -t 6:00:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $SCRATCH/src/optimize_simple_network

ibrun -n 2240 python3 -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
    --output-dir=data --pop_size=200 --max_iter=50 --path_length=3 --disp --procs_per_worker=112
EOT

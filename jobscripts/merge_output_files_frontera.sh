#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=merge_output_files_"$2"_"$DATE"
export MERGE_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/optimize_simple_network/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/optimize_simple_network/$JOB_NAME.%j.e
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:30:00
#SBATCH --mail-user=neurosutras@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK/optimize_simple_network

ibrun -n 1 python3 merge_output_files.py --merge-file-path=$MERGE_FILE_PATH --verbose \
  --output-dir=$SCRATCH/data/optimize_simple_network
EOT

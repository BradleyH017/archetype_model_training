#!/bin/bash
#BSUB -o logs/nextflow_master-%J-output.log
#BSUB -e logs/nextflow_master-%J-error.log
#BSUB -q oversubscribed
#BSUB -G team152
#BSUB -n 1
#BSUB -M 10000
#BSUB -a "memlimit=True"
#BSUB -R "select[mem>10000] rusage[mem=10000] span[hosts=1]"
#BSUB -J nextflow_master

mkdir -p logs work

module load cellgen/singularity
module load nextflow-23.10.0

# Nextflow settings
export NXF_OPTS="-Xms25G -Xmx25G"
# Uncomment this if get strange bus errors
# export NXF_OPTS="${NXF_OPTS} -Dleveldb.mmap=false" # No resume functionality
export NXF_HOME=$(pwd)
export NXF_WORK="${NXF_HOME}/work"
export NXF_TEMP="${NXF_HOME}/nextflow_temp"
export NXF_CONDA_CACHEDIR="${NXF_HOME}/nextflow_conda"
export NXF_SINGULARITY_CACHEDIR="${NXF_HOME}/cache_singularity"

# Farm specific settings
#export QT_QPA_PLATFORM='offscreen'
export LSB_DEFAULT_JOBGROUP="/${USER}/nf"
# export LSB_DEFAULTGROUP="sc-eqtl-ibd"
# export LSB_DEFAULTGROUP="cnv_15x"
export LSB_DEFAULTGROUP="team152"
export PYTHONHASHSEED=0
mkdir -p $NXF_TEMP

nextflow run workflow/main.nf \
    -profile sanger \
    -resume \
    -work-dir ${PWD}/work \
    -c conf/training.config \
    -with-report logs/report.html \
    -with-timeline logs/timeline.html

# Submit with:
# bash submit_nextflow_BH.sh
#
# To use full gene list instead of the test set, change in nextflow.config:
#   discovery_dataset_list = "input/all_input.txt"
#
# To override any param at runtime without editing the config:
#   nextflow run workflow/main.nf -resume --discovery_dataset_list input/all_input.txt

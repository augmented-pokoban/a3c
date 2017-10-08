#!/bin/sh
### General options
### –- specify queue --
#PBS -q hpc
### -- set the job Name --
#BSUB -N A3C_breakout_v0
# –- number of processors/cores/nodes --
#PBS -l nodes=1:ppn=10
### -- set walltime limit: hh:mm --
#PBS -l walltime=24:00:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##PBS -M arydbirk@gmail.com
### -- send notification at start --
#PBS -m abe
### -- Select the resources: 2 gpus in exclusive process mode --
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#PBS -o Output_a3c_breakout_v1_%N.out
#PBS -e Error_a3c_breakout_v1_%N.err

module load numpy/1.11.2-python-2.7.12-openblas-0.2.15_ucs4
module load scipy/scipy-0.18.1-python-2.7.12_ucs4
source /appl/tensorflow/1.1cpu/bin/activate

echo "Loaded modules"

p





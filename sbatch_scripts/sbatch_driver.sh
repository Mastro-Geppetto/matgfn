#!/bin/bash
export NETWORK=/users/sgpchatt/code/OTHER/zeo++-0.3/network

for run_name in 'asc' 'ats' 'cdl-e' 'cdz-e' 'dmg' 'dnq' 'eft' 'ffc' 'fso' 'tff' 'tsg' 'urj' ; do
  echo "run name : $run_name"
  export RUN_NAME=$run_name
  # one core one node 2 days time limit
  sbatch -J gen_$RUN_NAME -N 1 -n 2 -t 2-23:55:00 -o gen_$RUN_NAME.%u.%N.%j.out -e gen_$RUN_NAME.%u.%N.%j.err ./sbatch_gen_with_edge.sh
  sbatch -J genwo_$RUN_NAME -N 1 -n 2 -t 2-23:55:00 -o genwo_$RUN_NAME.%u.%N.%j.out -e genwo_$RUN_NAME.%u.%N.%j.err ./sbatch_gen_withOUT_edge.sh
done
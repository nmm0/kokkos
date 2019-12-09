#!/bin/bash

LOOP_REPEAT=1
if [ "$1" != "" ]; then
   LOOP_REPEAT=$1
   echo Repeat loop = $LOOP_REPEAT
fi

# BytesAndFlops
pushd ./build/bytes_and_flops

USE_CUDA=`grep "ENABLE_CUDA" KokkosCore_config.h | wc -l`

if [[ ${USE_CUDA} > 0 ]]; then
  BAF_EXE=bytes_and_flops.cuda
  TEAM_SIZE=256
else
  BAF_EXE=bytes_and_flops.host
  TEAM_SIZE=1
fi

BAF_PERF_1=`./${BAF_EXE} 2 100000 1024 1 1 1 1 ${TEAM_SIZE} 6000 | awk '{print $12/174.5}'`
BAF_PERF_2=`./${BAF_EXE} 2 100000 1024 16 1 8 64 ${TEAM_SIZE} 6000 | awk '{print $14/1142.65}'`

echo "BytesAndFlops: ${BAF_PERF_1} ${BAF_PERF_2}"
popd

# MiniMD
pushd ./build/miniMD
rm minimd.1
rm minimd.2
cp ../../miniMD/kokkos/Cu_u6.eam ./
for i in $( seq 0 ${LOOP_REPEAT} )
do
  mpiexec -n 1 ./miniMD --half_neigh 0 -s 60 --ntypes 1 -t ${OMP_NUM_THREADS} -i ../../miniMD/kokkos/in.eam.miniMD > md_perf1.out
  mpiexec -n 1 ./miniMD --half_neigh 0 -s 20 --ntypes 1 -t ${OMP_NUM_THREADS} -i ../../miniMD/kokkos/in.eam.miniMD  > md_perf2.out

  cat md_perf1.out | grep PERF_SUMMARY | awk '{print $10/21163341}' >> minimd.1
  cat md_perf2.out | grep PERF_SUMMARY | awk '{print $10/13393417}' >> minimd.2
done
TOT_PERF_1=0
let loop_cnt=0
while read -r result_one; do
   TOT_PERF_1=$(echo ${TOT_PERF_1} + ${result_one} | bc)
   let loop_cnt+=1
done < ./minimd.1

MD_AVG_PERF_1=$(echo "scale=7; ${TOT_PERF_1} / ${loop_cnt}" | bc)

TOT_PERF_2=0
let loop_cnt=0
while read -r result_one; do
   TOT_PERF_2=$(echo ${TOT_PERF_2} + ${result_one} | bc)
   let loop_cnt+=1
done < ./minimd.2

MD_AVG_PERF_2=$(echo "scale=7; ${TOT_PERF_2} / ${loop_cnt}" | bc)
echo "miniMD: ${MD_AVG_PERF_1} ${MD_AVG_PERF_2} ${loop_cnt}"
popd

# MiniFE
pushd ./build/miniFE
rm minife.1
rm minife.2
for i in $( seq 0 ${LOOP_REPEAT} )
do
  rm *.yaml
  ./miniFE.x -nx 100 &> /dev/null
  FE_PERF_1=`grep "CG Mflop" *.yaml | awk '{print $4/14174}'`
  rm *.yaml
  ./miniFE.x -nx 50 &> /dev/null
  FE_PERF_2=`grep "CG Mflop" *.yaml | awk '{print $4/11897}'`
  echo ${FE_PERF_1} >> minife.1
  echo ${FE_PERF_2} >> minife.2
done
TOT_PERF_1=0
let loop_cnt=0
while read -r result_one; do
   TOT_PERF_1=$(echo ${TOT_PERF_1} + ${result_one} | bc)
   let loop_cnt+=1
done < ./minife.1

FE_AVG_PERF_1=$(echo "scale=7; ${TOT_PERF_1} / ${loop_cnt}" | bc)

TOT_PERF_2=0
let loop_cnt=0
while read -r result_one; do
   TOT_PERF_2=$(echo ${TOT_PERF_2} + ${result_one} | bc)
   let loop_cnt+=1
done < ./minife.2

FE_AVG_PERF_2=$(echo "scale=7; ${TOT_PERF_2} / ${loop_cnt}" | bc)
echo "MiniFE: ${FE_AVG_PERF_1} ${FE_AVG_PERF_2} ${loop_cnt}"

popd

PERF_RESULT=`echo "${BAF_PERF_1} ${BAF_PERF_2} ${MD_AVG_PERF_1} ${MD_AVG_PERF_2} ${FE_AVG_PERF_1} ${FE_AVG_PERF_2}" | awk '{print ($1+$2+$3+$4+$5+$6)/6}'`
echo "Total Result: " ${PERF_RESULT}

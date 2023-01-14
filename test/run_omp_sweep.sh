#!/bin/bash

# NOTE: run this script from your SparTen OpenMP build directory

DATA="test/data/cpapr_test_100x100x100_1e+06/tensor.txt"
MAX_THREADS=32
MAX_OUTER_ITER=20
MAX_INNER_ITER=10
TOLERANCE=1e-10
SEED=12345

echo "=============================================================================="
echo "SparTen OpenMP Performance Tests"
echo "Data: $DATA"
echo "=============================================================================="

for METHOD in Multiplicative-Update Quasi-Newton Damped-Newton
do
    for PLACE in threads cores sockets
    do
	export OMP_PLACES=${PLACE}
	for BIND in spread close
	do
	    export OMP_PROC_BIND=${BIND}
	    for ((THREADS=1; THREADS <= ${MAX_THREADS} ; THREADS=THREADS*2))
	    do
		export OMP_NUM_THREADS=${THREADS}
		./bin/Sparten_main \
		    --nComponent 5 \
		    --input-file ${DATA} \
		    --solver ${METHOD} \
		    --maxOuterIter ${MAX_OUTER_ITER} \
		    --maxInnerIter ${MAX_INNER_ITER} \
		    --tolerance ${TOLERANCE} \
		    --randomSeed ${SEED} \
		    2>&1 > tmp.out
		thread_pool=`grep thread_pool_topology tmp.out | cut -d[ -f2 | cut -dx -f2 | tr -d [:space:]`
		elapsed_time=`grep "Elapsed time" tmp.out | cut -d":" -f2 | tr -d [:space:]`
		echo "${METHOD},${PLACE},${BIND},${THREADS},${thread_pool},${elapsed_time}"
	    done
	done
    done
done

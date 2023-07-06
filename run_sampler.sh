#! /bin/bash

CONTAINER=docker
# CONTAINER=podman
NAME=run_sampler

${CONTAINER} start ${NAME} &>/dev/null
CONTAINER_EXISTS=$?

set -e

cd problem/
if [ ${CONTAINER_EXISTS} -ne 0 ]; then
    ${CONTAINER} run --name ${NAME} -dit -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable
    ${CONTAINER} exec ${NAME} pip install --user joblib
fi

for RATE in 1 2; do
    for VARIANCE in 2_0 4_0; do
        FILE_PATH="darcy_lognormal_rate_${RATE}/${VARIANCE}"
        ${CONTAINER} exec ${NAME} python3 parametric_pde_sampling/compute_samples.py -b 100 ${FILE_PATH} 20000
        ${CONTAINER} exec ${NAME} python3 parametric_pde_sampling/compute_functional.py -f identity ${FILE_PATH}
    done
done

${CONTAINER} exec ${NAME} python3 parametric_pde_sampling/compute_samples.py -b 100 darcy_rauhut 20000
${CONTAINER} exec ${NAME} python3 parametric_pde_sampling/compute_functional.py -f integral darcy_rauhut

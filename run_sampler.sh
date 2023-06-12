#! /bin/bash

# CONTAINER=docker
CONTAINER=podman
NAME=run_sampler

${CONTAINER} start ${NAME} &>/dev/null
CONTAINER_EXISTS=$?

set -e

cd problem/
if [ ${CONTAINER_EXISTS} -ne 0 ]; then
    ${CONTAINER} run --name ${NAME} -dit -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable
    ${CONTAINER} exec ${NAME} pip install --user joblib
fi

for VARIANCE in 2 5 10; do
    ${CONTAINER} exec ${NAME} python3 parametric_pde_sampling/compute_samples.py -b 100 darcy_lognormal_${VARIANCE} 20000
    ${CONTAINER} exec ${NAME} python3 parametric_pde_sampling/compute_functional.py -f identity darcy_lognormal_${VARIANCE}
done

${CONTAINER} exec ${NAME} python3 parametric_pde_sampling/compute_samples.py -b 100 darcy_rauhut 20000
${CONTAINER} exec ${NAME} python3 parametric_pde_sampling/compute_functional.py -f integral darcy_rauhut

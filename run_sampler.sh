#! /bin/bash

cd problem/
NAME=run_sampler
EXEC="docker exec ${NAME}"
docker start ${NAME} &>/dev/null
if [ $? -ne 0 ]; then
    docker run --name ${NAME} -dit -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:current
    ${EXEC} pip install --user joblib
fi

for VARIANCE in 2 5 10; do
    ${EXEC} python3 parametric_pde_sampling/compute_samples.py -b 100 darcy_lognormal_${VARIANCE} 20000
    ${EXEC} python3 parametric_pde_sampling/compute_functional.py -f identity darcy_lognormal_${VARIANCE}
done

${EXEC} python3 parametric_pde_sampling/compute_samples.py -b 100 darcy_rauhut 20000
${EXEC} python3 parametric_pde_sampling/compute_functional.py -f integral darcy_rauhut

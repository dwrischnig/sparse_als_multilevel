#! /bin/bash

for t in 50 100 500 1000; do
    python run.py darcy_rauhut ${t} 1000 -q integral -a sals
    python run.py darcy_rauhut ${t} 1000 -q integral -a tensap
    python run.py darcy_rauhut ${t} 1000 -q integral -a ssals
done 

for k in 2_0 4_0; do
    for t in 50 100 500 1000; do
        python run.py darcy_lognormal_${k} ${t} 1000 -q pod_mode -a sals
        python run.py darcy_lognormal_${k} ${t} 1000 -q pod_mode -a tensap
        python run.py darcy_lognormal_${k} ${t} 1000 -q pod_mode -a ssals
    done 
done

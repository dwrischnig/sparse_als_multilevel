#! /bin/bash

for t in 50 100 500 1000; do
    python run.py problem/darcy_rauhut ${t} 1000 -q integral -a sals
    python run.py problem/darcy_rauhut ${t} 1000 -q integral -a tensap
    python run.py problem/darcy_rauhut ${t} 1000 -q integral -a ssals
done 

for RATE in 1 2; do
    for VARIANCE in 2_0 4_0; do
        for t in 50 100 500 1000; do
            python run.py problem/darcy_lognormal_rate_${RATE}/${VARIANCE} ${t} 1000 -q pod_mode -a sals
            python run.py problem/darcy_lognormal_rate_${RATE}/${VARIANCE} ${t} 1000 -q pod_mode -a tensap
            python run.py problem/darcy_lognormal_rate_${RATE}/${VARIANCE} ${t} 1000 -q pod_mode -a ssals
        done
    done
done

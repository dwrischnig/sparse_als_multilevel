#! /bin/bash

python table_plots.py problem/darcy_rauhut 

for RATE in 1 2; do
    for VARIANCE in 2_0 4_0; do
        python table_plots.py problem/darcy_lognormal_rate_${RATE}/${VARIANCE}
    done
done

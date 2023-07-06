#! /bin/bash

echo "================================================================"
echo "Create borehole parameters and data"
echo "----------------------------------------------------------------"
python problem/borehole.py 20000
echo "================================================================"

echo "================================================================"
echo "Run borehole experiments"
echo "----------------------------------------------------------------"
for t in 50 100 500 1000; do
    python run.py problem/borehole ${t} 1000 -q identity -a sals
    python run.py problem/borehole ${t} 1000 -q identity -a tensap
    python run.py problem/borehole ${t} 1000 -q identity -a ssals
done 
echo "================================================================"

echo "================================================================"
echo "Create borehole tables"
echo "----------------------------------------------------------------"
python table_plots.py problem/borehole/identity_d20_s1000
echo "================================================================"
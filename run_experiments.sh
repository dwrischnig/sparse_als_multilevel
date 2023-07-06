for t in 50 100 500 1000; do
    python run.py darcy_rauhut ${t} 1000 -a sals
    python run.py darcy_rauhut ${t} 1000 -a tensap
    python run.py darcy_rauhut ${t} 1000 -a ssals
done 


for k in 1_5 2_0 3_0; do
    for t in 50 100 500 1000; do
        python run.py darcy_lognormal_${k} ${t} 1000 -a sals
        python run.py darcy_lognormal_${k} ${t} 1000 -a tensap
        python run.py darcy_lognormal_${k} ${t} 1000 -a ssals
    done 
done

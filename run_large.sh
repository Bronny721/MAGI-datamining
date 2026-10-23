#!/bin/sh
# Large datasets on an 8GB GPU: smaller training batch (README uses 2048), 60-minute training budget as in the paper
PY=${PY:-../.venv/Scripts/python}
mkdir -p logs results
$PY -u train_sage.py --runs 1 --dataset ogbn-arxiv --batchsize 1024 --max_duration 60 --kmeans_device cpu --kmeans_batch -1 --hidden '1024,256' --size '10,10' --wt 20 --wl 5 --tau 0.9 --ns 0.1 --lr 0.01 --epochs 400 --wd 0 --dropout 0 --num_workers 2 --result results/large_arxiv.csv > logs/large_arxiv.log 2>&1
echo "arxiv exit $?"
$PY -u train_sage.py --runs 1 --dataset Reddit --batchsize 1024 --max_duration 60 --kmeans_device cpu --kmeans_batch -1 --hidden '1024,256' --size '10,10' --wt 20 --wl 5 --tau 0.5 --ns 0.5 --lr 0.01 --epochs 400 --wd 0 --dropout 0 --num_workers 0 --result results/large_reddit.csv > logs/large_reddit.log 2>&1
echo "reddit exit $?"
$PY -u train_sage.py --runs 1 --dataset ogbn-products --batchsize 512 --test_batch 2000 --max_duration 60 --kmeans_device cuda --kmeans_batch 20000 --hidden '1024,1024,256' --size '10,10,10' --wt 20 --wl 4 --tau 0.9 --ns 0.1 --lr 0.01 --epochs 400 --wd 0 --dropout 0 --num_workers 0 --result results/large_products.csv > logs/large_products.log 2>&1
echo "products exit $?"
echo LARGE DONE

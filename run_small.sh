#!/bin/sh
PY=${PY:-../.venv/Scripts/python}
$PY -X utf8 train_gcn.py --runs 10 --dataset Cora --hidden 512 --wt 100 --wl 2 --tau 0.3 --ns 0.5 --lr 0.0005 --epochs 400 --wd 1e-3 > logs/cora.log 2>&1
$PY -X utf8 train_gcn.py --runs 10 --dataset Citeseer --hidden 1024,512 --wt 100 --wl 3 --tau 0.9 --ns 0.5 --lr 0.0001 --epochs 400 --wd 5e-4 > logs/citeseer.log 2>&1
$PY -X utf8 train_gcn.py --runs 10 --dataset Photo --hidden 512 --wt 100 --wl 3 --tau 0.5 --ns 0.5 --lr 0.0005 --epochs 400 --wd 1e-3 > logs/photo.log 2>&1
$PY -X utf8 train_gcn.py --runs 10 --dataset Computers --hidden 1024,512 --wt 100 --wl 3 --tau 0.9 --ns 0.1 --lr 0.0005 --epochs 400 --wd 1e-3 > logs/computers.log 2>&1
echo ALLDONE

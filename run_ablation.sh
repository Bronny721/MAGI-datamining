#!/bin/sh
# Ablation: baseline MAGI vs. improvements, same seeds (0..RUNS-1) for every config.
PY="${PY:-../.venv/Scripts/python} -X utf8"
RUNS=${RUNS:-5}
DATASETS=${DATASETS:-"Cora Citeseer Photo"}
CONFIGS=${CONFIGS:-"base count deg deg_count"}
for ds in $DATASETS; do
  case $ds in
    Cora)      P="--hidden 512 --wt 100 --wl 2 --tau 0.3 --ns 0.5 --lr 0.0005 --epochs 400 --wd 1e-3" ;;
    Citeseer)  P="--hidden 1024,512 --wt 100 --wl 3 --tau 0.9 --ns 0.5 --lr 0.0001 --epochs 400 --wd 5e-4" ;;
    Photo)     P="--hidden 512 --wt 100 --wl 3 --tau 0.5 --ns 0.5 --lr 0.0005 --epochs 400 --wd 1e-3" ;;
    Computers) P="--hidden 1024,512 --wt 100 --wl 3 --tau 0.9 --ns 0.1 --lr 0.0005 --epochs 400 --wd 1e-3" ;;
  esac
  for cfg in $CONFIGS; do
    case $cfg in
      base)      O="" ;;
      count)     O="--pos_weight count" ;;
      deg)       O="--deg_neg" ;;
      deg_count) O="--deg_neg --pos_weight count" ;;
      deg05)     O="--deg_neg --deg_beta 0.5" ;;
      pdeg05)    O="--pos_deg --pos_beta 0.5" ;;
      pdeg1)     O="--pos_deg --pos_beta 1.0" ;;
      sp10)      O="--sp_warmup 200 --sp_drop 0.1" ;;
      sp20)      O="--sp_warmup 200 --sp_drop 0.2" ;;
      sp30)      O="--sp_warmup 200 --sp_drop 0.3" ;;
      sp30w100)  O="--sp_warmup 100 --sp_drop 0.3" ;;
      sp50w100)  O="--sp_warmup 100 --sp_drop 0.5" ;;
      all)       O="--pos_thresh all" ;;
      se95)      O="--se_warmup 100 --se_tau 0.95" ;;
      se90)      O="--se_warmup 100 --se_tau 0.9" ;;
      se98)      O="--se_warmup 100 --se_tau 0.98" ;;
    esac
    $PY train_gcn.py --runs $RUNS --seed 0 --dataset $ds $P $O --verbose '' --result results/${TAG}${ds}_${cfg}.csv > logs/abl_${TAG}${ds}_${cfg}.log 2>&1
    echo "$ds $cfg $(grep '^mean' logs/abl_${TAG}${ds}_${cfg}.log) $(grep -c Traceback logs/abl_${TAG}${ds}_${cfg}.log) errors"
  done
done
echo ABLATION DONE

# MAGI 復現與改良（資料探勘期末專題）

本 repo 以 MAGI 官方程式碼為基礎，完成以下工作：

1. **復現**：在 4 個小資料集（Cora、Citeseer、Amazon-Photo、Amazon-Computers）與 3 個大資料集（ogbn-arxiv、Reddit、ogbn-products）上重跑 MAGI。
2. **改良**：針對 MAGI 的正／負樣本選取方式提出多種變體（見下方〈改良變體〉），以相同 seed 做消融實驗與 Wilcoxon 成對顯著性檢定。

- 論文：Yunfei Liu, Jintang Li, et al. *Revisiting Modularity Maximization for Graph Clustering: A Contrastive Learning Perspective.* KDD 2024. [arXiv:2406.14288](https://arxiv.org/abs/2406.14288)
- 官方程式碼：<https://github.com/EdisonLeeeee/MAGI>（本 repo 的 git 歷史保留了官方 commit，我們的修改可以用 `git diff 6313854` 查看）

<div align="center">
  <img width="600" src="imgs/framework.png"/>
</div>

---

## 1. 實驗環境

| 項目 | 規格 |
|---|---|
| 作業系統 | Windows 11 Home |
| GPU | NVIDIA GeForce RTX 4060 Ti（8 GB），driver 610.88，CUDA 11.8 |
| RAM | 32 GB |
| Python | 3.13.4 |
| 主要套件 | torch 2.7.1+cu118、torch-geometric 2.8.1、torch-scatter 2.1.2、torch-sparse 0.6.18、torch-cluster 1.6.3、ogb 1.3.6、scikit-learn 1.7.1、scipy 1.16.0、numpy 2.2.6 |

完整版本列在 `requirements.txt`。官方 README 使用的是 torch 1.10 / PyG 2.0.4，我們改用新版套件，並修正了因此產生的相容性問題（見〈與官方程式碼的差異〉）。

## 2. 安裝

```bash
git clone <本 repo 網址>
cd MAGI

# 建立虛擬環境（腳本預設的 Python 路徑是 ../.venv，也可以用 PY 環境變數指定，見第 4 節）
python -m venv ../.venv
# Windows
../.venv/Scripts/activate
# Linux / macOS
# source ../.venv/bin/activate

# 1) PyTorch（CUDA 11.8）
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
# 2) PyG 的 C++ 擴充套件，需要從 PyG 的 wheel 網址安裝
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
# 3) 其餘套件
pip install torch-geometric==2.8.1 ogb==1.3.6 numpy==2.2.6 scipy==1.16.0 scikit-learn==1.7.1 munkres==2.0.0 matplotlib==3.10.3 tqdm==4.67.1
```

`scikit-learn-intelex` 是選用套件：有安裝的話 k-means 會比較快，沒有也能正常執行。

## 3. 資料集

所有資料集在第一次執行時都會**自動下載**到 `data/`，不需要手動準備（Cora 和 Citeseer 已經包含在 repo 中）。

| 資料集 | 節點數 | 邊數（依論文 Table 1） | 特徵維度 | 類別數 | 來源 | 下載位置 |
|---|---|---|---|---|---|---|
| Cora | 2,708 | 5,278 | 1,433 | 7 | PyG `Planetoid`（[Planetoid splits](https://github.com/kimiyoung/planetoid)） | `data/Cora` |
| Citeseer | 3,327 | 4,614 | 3,703 | 6 | PyG `Planetoid` | `data/Citeseer` |
| Amazon-Photo | 7,650 | 119,081 | 745 | 8 | PyG `Amazon`（[Shchur et al.](https://github.com/shchur/gnn-benchmark)） | `data/Photo` |
| Amazon-Computers | 13,752 | 491,722 | 767 | 10 | PyG `Amazon` | `data/Computers` |
| ogbn-arxiv | 169,343 | 1,166,243 | 128 | 40 | [OGB](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) | `data/OGB/ogbn_arxiv` |
| Reddit | 232,965 | 23,213,838 | 602 | 41 | PyG `Reddit`（[GraphSAGE](http://snap.stanford.edu/graphsage/)） | `data/Reddit` |
| ogbn-products | 2,449,029 | 61,859,140 | 100 | 47 | [OGB](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) | `data/OGB/ogbn_products` |

注意：大資料集的檔案很大（`data/` 全部下載完約 8 GB），ogbn-products 第一次處理需要較多 RAM（建議 32 GB 以上）。

## 4. 如何執行

以下指令都要在 repo 根目錄（`MAGI/`）下執行。Windows 請用 Git Bash 執行 `.sh` 腳本。

腳本預設使用 `../.venv/Scripts/python`（Windows venv 的路徑）。若 Python 在其他位置，用 `PY` 環境變數指定，例如：

```bash
PY=python sh run_small.sh                    # 用目前環境的 python
PY=../.venv/bin/python sh run_small.sh       # Linux venv
```

Windows 上執行 Python 建議加 `-X utf8`，避免編碼錯誤（腳本裡已經加了）。

### 4.1 小資料集復現（`train_gcn.py`）

```bash
mkdir -p logs results
sh run_small.sh
```

會用官方 README 的超參數在 Cora / Citeseer / Photo / Computers 各跑 10 次（隨機 seed），結果寫在 `logs/{cora,citeseer,photo,computers}.log`。單一資料集也可以直接執行，例如：

```bash
python -X utf8 train_gcn.py --runs 10 --dataset Cora --hidden 512 --wt 100 --wl 2 --tau 0.3 --ns 0.5 --lr 0.0005 --epochs 400 --wd 1e-3
```

在 RTX 4060 Ti 上，每個資料集 10 次約數分鐘到十幾分鐘。

### 4.2 改良方法的消融實驗（`run_ablation.sh`）

`run_ablation.sh` 會對指定的資料集與變體，用**相同的 seed（0, 1, …, RUNS-1）**執行，方便做成對檢定。透過環境變數控制：

| 變數 | 預設值 | 說明 |
|---|---|---|
| `RUNS` | `5` | 每個設定跑幾次 |
| `DATASETS` | `"Cora Citeseer Photo"` | 要跑的資料集 |
| `CONFIGS` | `"base count deg deg_count"` | 要跑的變體（名稱見 4.4） |
| `TAG` | 空 | 輸出檔名前綴 |

輸出：每次的結果寫在 `results/${TAG}${資料集}_${變體}.csv`，log 寫在 `logs/abl_${TAG}${資料集}_${變體}.log`。

```bash
# 早期消融（5 runs）
sh run_ablation.sh
CONFIGS="deg05 pdeg05 pdeg1 sp10 sp20 sp30" DATASETS="Cora Citeseer" sh run_ablation.sh
CONFIGS="pdeg05 pdeg1" DATASETS="Photo" sh run_ablation.sh

# 最終比較（10 runs）
RUNS=10 TAG=final_ CONFIGS="base se95" DATASETS="Cora Citeseer Computers Photo" sh run_ablation.sh
RUNS=10 TAG=final_ CONFIGS="se90 se98 all sp20 sp30w100 sp50w100" DATASETS="Photo" sh run_ablation.sh
```

### 4.3 大資料集（`train_sage.py`）

```bash
sh run_large.sh
```

依序執行 ogbn-arxiv、Reddit、ogbn-products，各 1 次，訓練時間上限 60 分鐘（與論文設定相同）。結果寫在 `results/large_{arxiv,reddit,products}.csv`。

因為 GPU 只有 8 GB，我們調整了以下參數（其餘與官方 README 相同）：

| 資料集 | 官方 batch size | 我們的 batch size | 其他調整 |
|---|---|---|---|
| ogbn-arxiv | 2048 | 1024 | — |
| Reddit | 2048 | 1024 | `--num_workers 0` |
| ogbn-products | 2048 | 512 | `--test_batch 2000`、GPU k-means `--kmeans_batch 20000`（官方 300000） |

ogbn-products 需要 60 分鐘訓練，加上推論與 GPU k-means（約 46 分鐘），總共約 2 小時。

### 4.4 改良變體與對應參數

`train_gcn.py` 與 `train_sage.py` 都支援以下新增參數（不加任何參數時就是原始 MAGI）：

| 變體名稱 | 參數 | 說明 |
|---|---|---|
| `base` | （無） | 原始 MAGI |
| `count` | `--pos_weight count` | 正樣本依 random walk 造訪次數加權（原本是等權重） |
| `deg` | `--deg_neg` | 負樣本依 degree 加權，對應 modularity 的 null model $k_ik_j/2m$，β = 1 |
| `deg05` | `--deg_neg --deg_beta 0.5` | 同上，權重為 (deg / 平均 deg)^0.5 |
| `deg_count` | `--deg_neg --pos_weight count` | `deg` + `count` |
| `pdeg05` / `pdeg1` | `--pos_deg --pos_beta 0.5` / `1.0` | 挑選正樣本時，門檻改用 degree-corrected null model |
| `sp10` / `sp20` / `sp30` | `--sp_warmup 200 --sp_drop 0.1` / `0.2` / `0.3` | Self-paced 正樣本去噪：第 200 epoch 起，每個 anchor 丟掉 embedding 最不相似的 x 比例正樣本 |
| `sp30w100` / `sp50w100` | `--sp_warmup 100 --sp_drop 0.3` / `0.5` | 同上，warmup 改為 100 epoch |
| `all` | `--pos_thresh all` | 依論文 Eq. 9，所有被 random walk 造訪過的節點都當正樣本（官方程式用平均值當門檻） |
| `se90` / `se95` / `se98` | `--se_warmup 100 --se_tau 0.9` / `0.95` / `0.98` | Embedding 引導的正樣本擴充：第 100 epoch 起，embedding cosine 相似度大於 τ 的節點也加入正樣本 |

其他新增的通用參數：

| 參數 | 說明 |
|---|---|
| `--seed S` | 第 i 次 run 使用 seed S+i（預設 -1 = 隨機） |
| `--result FILE.csv` | 把每次 run 與平均／標準差寫入 CSV |
| `--save_emb DIR` | （僅 `train_gcn.py`）儲存最後的 embedding，給 `eval_smooth.py` 使用 |

### 4.5 結果彙整與分析腳本

```bash
python -X utf8 summarize.py                 # 讀 results/*.csv，產生 results/summary.md 與 summary.csv（含 Wilcoxon 檢定）
python paired_test.py results/final_Photo_base.csv results/final_Photo_se95.csv   # 單獨做成對檢定
python pos_purity.py Cora Citeseer Photo    # 診斷：比較不同 null model 挑出的正樣本 label 純度
python train_gcn.py ... --save_emb emb/gpu && python eval_smooth.py --dataset Cora --emb_dir emb/gpu   # embedding 平滑後再分群
```

## 5. 主要結果

完整結果見 [`results/summary.md`](results/summary.md)，原始 log 在 `logs/`。

### 5.1 復現（ACC / NMI / ARI / F1，%）

| 資料集 | 論文數字 | 我們的復現（mean ± std） | Runs |
|---|---|---|---|
| Cora | 76.0 / 59.7 / 57.3 / 73.9 | 75.86±0.64 / 59.89±0.41 / 57.39±0.89 / 73.54±0.65 | 10 |
| Citeseer | 70.6 / 45.2 / 46.8 / 64.8 | 70.05±0.49 / 44.40±0.64 / 45.93±0.65 / 65.07±0.59 | 10 |
| Photo | 79.0 / 71.6 / 61.5 / 72.9 | 79.24±0.35 / 70.40±0.36 / 62.45±0.58 / 72.71±0.23 | 10 |
| Computers | 62.0 / 59.2 / 46.2 / 57.4 | 61.97±1.10 / 58.95±1.04 / 44.28±1.15 / 60.52±1.47 | 10 |
| ogbn-arxiv | 38.8 / 46.9 / 31.0 / 26.6 | 40.84 / 47.44 / 32.95 / 27.52 | 1 |
| Reddit | 91.1 / 87.5 / 90.7 / 85.3 | 92.46 / 88.38 / 91.38 / 87.50 | 1 |
| ogbn-products | 42.5 / 55.1 / 21.5 / 27.6 | 38.56 / 53.65 / 19.78 / 25.45 | 1 |

小資料集的結果與論文差距都在約 1 個百分點以內。ogbn-products 低約 4 個百分點，可能和 batch size 縮小（2048 → 512）以及 60 分鐘內能完成的 epoch 數較少有關。

### 5.2 改良變體（最終比較，10 runs，相同 seed，ACC %）

| 資料集 | base | se95 | ΔACC（勝場） | Wilcoxon p（ACC） |
|---|---|---|---|---|
| Cora | 76.10±0.62 | 76.00±0.65 | −0.10（6/10） | 0.752 |
| Citeseer | 70.13±0.38 | 70.26±0.38 | +0.13（8/10） | **0.035** |
| Computers | 62.15±1.15 | 62.46±1.07 | +0.30（7/10） | 0.232 |
| Photo | 78.75±0.70 | 78.99±0.61 | +0.24（9/10） | **0.014** |

Photo 上的其他變體：se90 78.50、se98 78.69、all 78.31、sp20 78.15、sp30w100 77.21、sp50w100 73.15（sp50w100 在四個指標上都顯著變差，p = 0.002）。

- `se95` 在 Citeseer 與 Photo 上有小幅但統計顯著的提升（ACC 與 ARI 皆 p < 0.05）；Cora、Computers 則無顯著差異。Photo 一共比較了 7 個變體，若做 Bonferroni 校正（門檻 0.05/7 ≈ 0.007），p = 0.014 就不再顯著。
- Self-paced 去噪（`sp`）與 `all` 在 Photo 上顯著變差。
- `deg` 類變體在 Cora 上看起來有幫助（+0.97），但在 Citeseer、Photo 上變差；只有 3～5 runs，不足以下結論。
- 早期消融（5 runs）與最終比較（10 runs）的 base 是不同批次，兩段的數字不要互相比較。

## 6. 檔案結構

```
MAGI/
├── train_gcn.py           # 小資料集（full-batch GCN encoder）訓練與評估
├── train_sage.py          # 大資料集（mini-batch GraphSAGE encoder）訓練與評估
├── magi/
│   ├── model.py           # Encoder 與 MAGI 模型
│   ├── loss.py            # 對比損失（新增正樣本加權、degree 加權負樣本）
│   ├── utils.py           # random walk 正樣本選取、se/sp 正樣本更新、分群評估
│   ├── neighbor_sampler.py# 鄰居採樣（新增向量化版本）
│   ├── batch_kmeans_cuda.py
│   └── clustering_metric.py
├── run_small.sh           # 4.1 小資料集復現
├── run_ablation.sh        # 4.2 消融 / 最終比較
├── run_large.sh           # 4.3 大資料集
├── summarize.py           # 產生 results/summary.md
├── paired_test.py         # Wilcoxon 成對檢定
├── pos_purity.py          # 正樣本 label 純度診斷
├── eval_smooth.py         # embedding 平滑後分群
├── results/               # 每個實驗的 CSV 與 summary.md
├── logs/                  # 每個實驗的完整 log
└── data/                  # 資料集（自動下載；只有 Cora、Citeseer 在 repo 內）
```

## 7. 與官方程式碼的差異

- **新增改良變體**：4.4 節列出的所有參數（`magi/loss.py`、`magi/utils.py`、`train_gcn.py`、`train_sage.py`）。
- **可重現性**：新增 `--seed`，讓不同變體使用相同 seed，可以做成對檢定；新增 `--result` 輸出 CSV。
- **新版套件相容性**：配合 torch 2.7 / PyG 2.8 / numpy 2 調整（例如 PyTorch 2.6 之後 `torch.load` 預設只載入權重，需要把 PyG 的資料類別加入允許清單，OGB 資料集才能載入）。
- **大資料集在 8 GB GPU 上執行**：新增 `--test_batch`（推論分批）、向量化的鄰居採樣器（舊版可用 `--legacy_sampler`）。

## 8. 引用

```bib
@inproceedings{magi,
  title     = {Revisiting Modularity Maximization for Graph Clustering: A Contrastive Learning Perspective},
  author    = {Yunfei Liu and Jintang Li and Yuehe Chen and Ruofan Wu and Baokun Wang and Jing Zhou and Sheng Tian and Shuheng Shen and Xing Fu and Changhua Meng and Weiqiang Wang and Liang Chen},
  booktitle = {KDD},
  publisher = {ACM},
  year      = {2024},
}
```

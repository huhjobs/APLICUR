# APLICUR: Adaptively Preconditioned Iterative Solver for Linear Least-Squares

This repository contains the MATLAB implementation accompanying the paper:

> **Adaptively Preconditioned LSQR from One Small Sketch**  
> [[arXiv:2604.05065]](https://arxiv.org/abs/2604.05065)

---

## Overview

APLICUR solves large-scale regularized linear least-squares problems of the form

$$\min_x \|Ax - b\|_2^2 + \mu^2 \|x\|_2^2$$

by adaptively building a CUR-based preconditioner and applying it within an iterative LSQR solver.

---

## Quick Start

See [`demo.m`](demo.m) for a minimal working example that generates a synthetic problem and runs APLICUR from scratch.

---

## Reproducing Experimental Results

All experiments from the paper can be reproduced using `batch_experiments.m`.

### Step 1: Configure `batch_experiments.m`

```matlab
dataset_path = '../datasets/datasetsubdirectory';  % path to dataset folder

% Filter datasets to run (uncomment one):
d = d(contains({d.name}, 'A'));       % single condition
% d = d((contains({d.name}, 'B') | ...
%        contains({d.name}, 'C')) & ...
%        contains({d.name}, 'D'));               % multiple conditions
% (leave d as-is to run all datasets)
```

### Step 2: Configure `run_experiments.m`

Select which algorithms to run by uncommenting the corresponding `run_*` calls:

```matlab
run_lsqr(...)       % LSQR baseline
run_aplicur(...)    % APLICUR (SVD-based)
run_aplicur(...)    % APLICUR (SVD-free), set 'ALGNAME' to 'aplicur_svdfree'
run_npcg(...)       % NPCG
run_bdp(...)        % Blendenpik
```

### Step 3: Run

```matlab
batch_experiments
```

Results are saved under `./experiments/results/datasetsubdirectory/<n>/` where `<n>` is the smallest unused positive integer, so repeated runs never overwrite previous results.

### Step 4: Plot Results

Configure and run `plot_results.m`:

```matlab
foldername = "datasetsubdirectory/1";   % results subfolder to plot
```

Figures are saved under `./experiments/resultplots/`.

---

## Repository Structure                                                                           
                  
```                                                                                            
.
├── README.md               
├── demo.m                         % Minimal working example — start here                         
├── algorithms/                                                                                   
│   ├── aplicur.m                  % Main APLICUR algorithm (SVD-based and SVD-free)
│   ├── aplicur_singleshot.m       % Single-shot / scheduled variant                              
│   └── lsqr_scheduled.m           % LSQR with scheduled restarts
├── baselines/                                                                                    
│   ├── blendenpik_matlab.m        % Blendenpik baseline
│   ├── lsqr_ethan_new.m           % LSQR baseline                                                
│   ├── npcg.m                     % Nystrom PCG baseline                                                
│   └── cg_ethan.m                 % CG baseline                                                  
├── experiments/                                                                                  
│   ├── batch_experiments.m        % Run experiments over multiple datasets                       
│   ├── run_experiments.m          % Run algorithms
│   ├── run_lsqr.m                 % Run LSQR
│   ├── run_aplicur.m              % Run APLICUR
│   ├── run_npcg.m                 % Run Nystrom PCG
│   ├── run_bdp.m                  % Run Blendenpik
│   ├── plot_results.m             % Plot and save figures                                        
│   ├── results/                   % Saved .mat outputs and terminal logs (auto-created)          
│   └── resultplots/               % Saved PDF figures (auto-created)                             
└── datasets/                                                                                     
  └── smallexampledata/            % Small test matrices (.mat)                                 
``` 

---

## Dataset Format

See [`datasets/README.md`](datasets/README.md) for the expected `.mat` file format and instructions on adding your own datasets.

---

## Dependencies

- MATLAB R2025b or later
- `lsqr_ethan_new.m` and `cg_ethan.m` are modified from [`my_lsqr.m`](https://github.com/eepperly/Stable-Randomized-Least-Squares/blob/main/code/mylsqr.m) and [`mycg.m`](https://github.com/eepperly/Stable-Randomized-Least-Squares/blob/main/code/mycg.m)
- [`sparsesign.m`](https://github.com/eepperly/Stable-Randomized-Least-Squares) for sparse embedding sketches

---

## Citation

If you use this code, please cite:

```bibtex
@article{huh2026aplicur,
  title   = {Adaptive Preconditioned Iterative Methods via CUR Decomposition for Regularized Least Squares},
  author={Jung Eun Huh and Coralia Cartis and Yuji Nakatsukasa},
  journal = {arXiv preprint arXiv:2604.05065},
  year    = {2026}
}
```
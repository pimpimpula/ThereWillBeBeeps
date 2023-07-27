# ThereWillBeBeeps

---

## Predictable sequential structure enhances auditory sensitivity in clinical audiograms

#####  Nadège Marin<sup>1,2</sup>, Grégory Gérenton<sup>1</sup>, Hadrien Jean<sup>3</sup>, Nihaad Paraouty<sup>3</sup>, Nicolas Wallaert<sup>3</sup>, Diane S. Lazard<sup>1,4</sup>, Luc H. Arnal<sup>1*</sup>, Keith B. Doelling<sup>1*</sup>

1.	Institut Pasteur, Université Paris Cité, Inserm UA06, Institut de l’Audition, F-75012 Paris, France
2.	Aix-Marseille University, Marseille, France
3.	My Medical Assistant SAS, Reims, France
4.	Institut Arthur Vernes, ENT department, F-75006 Paris, France

### Abstract

> Human hearing is highly sensitive and allows us to detect acoustic events at low levels and at great distances. 
> However, sensitivity is not only a function of the integrity of cochlear mechanisms, but also constrained by central 
> processes such as attention and expectation. While the effects of distraction and attentional orienting are generally 
> acknowledged, the extent to which probabilistic expectations influence sensitivity is not clear. 
> 
> Classical audiological assessment, commonly used to assess hearing sensitivity, does not distinguish between bottom-up
> sensitivity and top-down gain/filtering. In this study, we aim to decipher the influence of various types of 
> expectations on hearing sensitivity and how this information can be used to improve the assessment of sensitivity 
> to sound. 
> 
> Our results raise important critiques regarding common practices in the assessment of sound sensitivity, 
> both in fundamental research and in audiological clinical assessment.

---

### This repo

The project is organized as Jupyter notebooks, which contain the steps and output of the analysis. These notebooks call functions located in separate Python files.

#### Usage

Notebooks are meant to be run in increasing order (1_, 2_, etc), starting with preprocessing and then analysis. Each notebook contains detailed comments about what each section of code is doing.

The `preprocessing` and `analysis` directories contain subdirectories named `funcs`, which contain Python scripts for various functions that the notebooks call.

Note that you need an extra python script `API_access.py` which contains private access codes for the automated pure-tone audiometry API to recreate some of the dataframes in `data` > `dataframes`.

---

### Pipeline
    
#### Preprocessing

#### Analysis

---

### Repository structure:

```bash
.
├── [data](./data)
│   ├── [audiograms](./data/audiograms)
│   ├── [dataframes](./data/dataframes)
│   └── [raw_data](./data/raw_data)
│       └── one folder/participant
│           ├── Bayesian
│           ├── Continuous
│           ├── Cluster
│           └── 3AFC
│
├── [figures](./figures)
│
└── [scripts](./scripts)
    ├── __init__.py
    ├── [figure_params.py](./scripts/figure_params.py)
    ├── [stats.py](./scripts/stats.py)
    ├── [utils.py](./scripts/utils.py)
    ├── API_access.py *PRIVATE*
    │
    ├── [preprocessing](./scripts/preprocessing)
    │   ├── __init__.py
    │   │
    │   ├── [funcs](./scripts/preprocessing/funcs)
    │   │   ├── __init__.py
    │   │   ├── [fix_continuous_responses.py](./scripts/preprocessing/funcs/fix_continuous_responses.py)
    │   │   └── [resample_audiograms.py](./scripts/preprocessing/funcs/resample_audiograms.py)
    │   │
    │   ├── [1_3AFC_thresholds.ipynb](./scripts/preprocessing/1_3AFC_thresholds.ipynb)
    │   ├── [2_fix_continuous_responses.ipynb](./scripts/preprocessing/2_fix_continuous_responses.ipynb)
    │   └── [3_resample_audiograms.ipynb](./scripts/preprocessing/3_resample_audiograms.ipynb)
    │
    └── [analysis](./scripts/analysis)
        ├── __init__.py
        │
        ├── [funcs](./scripts/analysis/funcs)
        │   ├── __init__.py
        │   ├── [global_random_audiogram.py](./scripts/analysis/funcs/global_random_audiogram.py)
        │   ├── [linear_models.py](./scripts/analysis/funcs/linear_models.py)
        │   ├── [p50_analysis.py](./scripts/analysis/funcs/p50_analysis.py)
        │   ├── [performance_correlation.py](./scripts/analysis/funcs/performance_correlation.py)
        │   ├── [plots.py](./scripts/analysis/funcs/plots.py)
        │   └── [threshold_analysis.py](./scripts/analysis/funcs/threshold_analysis.py)
        │
        ├── [1_threshold_analysis.ipynb](./scripts/analysis/1_threshold_analysis.ipynb)
        ├── [2_global_random_audiogram.ipynb](./scripts/analysis/2_global_random_audiogram.ipynb)
        ├── [3_fig2C_p50_example_data.ipynb](./scripts/analysis/3_fig2C_p50_example_data.ipynb)
        ├── [4_p50_analysis.ipynb](./scripts/analysis/4_p50_analysis.ipynb)
        ├── [5_performance_correlation.ipynb](./scripts/analysis/5_performance_correlation.ipynb)
        ├── [6_clustering_paradigms.ipynb](./scripts/analysis/6_clustering_paradigms.ipynb)
        ├── [7_catch_trials.ipynb](./scripts/analysis/7_catch_trials.ipynb)
        ├── [8_linear_models.ipynb](./scripts/analysis/8_linear_models.ipynb)
        └── [SupplFig_example_data.ipynb](./scripts/analysis/SupplFig_example_data.ipynb)
```

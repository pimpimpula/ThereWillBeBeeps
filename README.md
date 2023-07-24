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

#### Repository structure

```
.
├── data
│   ├── audiograms
│   ├── dataframes
│   └── raw_data
│       └── one folder/participant
│           ├── Bayesian
│           ├── Continuous
│           ├── Cluster
│           └── 3AFC
├── figures
└── scripts
    ├── __init__.py
    ├── figure_params.py
    ├── stats.py
    ├── utils.py
    │
    ├── preprocessing
    │   ├── funcs
    │   │   ├── __init__.py
    │   │   ├── fix_continuous_responses.py
    │   │   └── resample_audiograms.py
    │   ├── __init__.py
    │   ├── 1_3AFC_thresholds.ipynb
    │   ├── 2_fix_continuous_responses.ipynb
    │   ├── 3_resample_audiograms.ipynb
    │   └── 4_get_catch_trials_data.ipynb
    │
    └── analysis
        ├── funcs
        │   ├── __init__.py
        │   ├── global_random_audiogram.py
        │   ├── p50_analysis.py
        │   ├── performance_correlation.py
        │   └── plots.py
        ├── __init__.py
        ├── 1_threshold_analysis.ipynb
        ├── 2_global_random_audiogram.ipynb
        ├── 3_fig2C_p50_example_data.ipynb
        ├── 4_p50_analysis.ipynb
        ├── 5_performance_correlation.ipynb
        ├── clustering_paradigms.py
        └── suppl_fig_example_data.ipynb

```


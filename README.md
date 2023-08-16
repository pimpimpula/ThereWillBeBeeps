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

The project is organized as Jupyter notebooks, which contain the steps and output of the analysis. 
These notebooks call functions located in separate Python files.

#### Usage

Notebooks are meant to be run in increasing order (1_, 2_, etc), starting with preprocessing and then analysis. Each notebook contains detailed comments about what each section of code is doing.

The [`preprocessing`](scripts%2Fpreprocessing) and [`analysis`](scripts%2Fanalysis) directories contain subdirectories named `funcs`, which contain Python scripts for various functions that the notebooks call.

Note: you need an extra python script `API_access.py` with private access codes for the automated pure-tone audiometry API to run some of the code (preprocessing 2-3, analysis 2).

The free-est way to run these notebooks is with jupyter notebook.
Here is one way to load this repo to run smoothly

**Clone code and move into folder**
```
git clone git@github:pimpimpula/ThereWillBeBeeps.git
cd ThereWillBeBeeps
```
**Create conda environment and install necessary packages**
```
conda create -n beepbeep python=3.10
conda activate beepbeep
pip install -r requirements.txt
conda install jupyter
```
**Include project directory as path in conda env and deactivate environment to go do something else.**
```
conda develop .
conda deactivate
```

Then when you are ready to run your notebooks just do the following while in the project directory:
```
conda activate beepbeep
jupyter notebook
```

#### Requirements

This repo relies on commonly used python packages for data analysis and visualization,
as well as the 'requests' module for communication with the pure-tone audiometry API.

See [`requirements.txt`](requirements.txt) for the detailed short list.

---

### Pipeline
    
#### Preprocessing

- Compute audiograms for the 3-AFC task  ──  [ 1_3AFC_thresholds](scripts%2Fpreprocessing%2F1_3AFC_thresholds.ipynb)
- Fix responses in the Continuous task ── [2_fix_continuous_responses](scripts%2Fpreprocessing%2F2_fix_continuous_responses.ipynb)
- Resample audiograms ── [3_resample_audiograms](scripts%2Fpreprocessing%2F3_resample_audiograms.ipynb)

#### Analysis

- Compare average thresholds ── [1_threshold_analysis](scripts%2Fanalysis%2F1_threshold_analysis.ipynb)
- Compute the global random audiogram ── [2_global_random_audiogram](scripts%2Fanalysis%2F2_global_random_audiogram.ipynb)
- Plot example data to visually explain p50 method ── [3_fig2C_p50_example_data](scripts%2Fanalysis%2F3_fig2C_p50_example_data.ipynb)
- Compare average p50 values ── [4_p50_analysis](scripts%2Fanalysis%2F4_p50_analysis.ipynb)
- Check performance correlation across conditions ── [5_performance_correlation](scripts%2Fanalysis%2F5_performance_correlation.ipynb)
- Cluster paradigms based on performance ── [6_clustering_paradigms](scripts%2Fanalysis%2F6_clustering_paradigms.ipynb)
- Compute false alarm rates ── [7_catch_trials](scripts%2Fanalysis%2F7_catch_trials.ipynb)
- Correlate performance with false alarm rate, musical sophistication & age ── [8_linear_models](scripts%2Fanalysis%2F8_linear_models.ipynb)

---

### Repository structure:

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
│
├── figures
│
├── README.md
├── requirements.txt
│
└── scripts
    ├── __init__.py
    ├── figure_params.py
    ├── stats.py
    ├── utils.py
    ├── API_access.py **** PRIVATE ****
    │
    ├── preprocessing
    │   ├── __init__.py
    │   │
    │   ├── funcs
    │   │   ├── __init__.py
    │   │   ├── fix_continuous_responses.py
    │   │   └── resample_audiograms.py
    │   │
    │   ├── 1_3AFC_thresholds.ipynb
    │   ├── 2_fix_continuous_responses.ipynb
    │   └── 3_resample_audiograms.ipynb
    │
    └── analysis
        ├── __init__.py
        │
        ├── funcs
        │   ├── __init__.py
        │   ├── global_random_audiogram.py
        │   ├── linear_models.py
        │   ├── p50_analysis.py
        │   ├── performance_correlation.py
        │   ├── plots.py
        │   └── threshold_analysis.py
        │
        ├── 1_threshold_analysis.ipynb
        ├── 2_global_random_audiogram.ipynb
        ├── 3_fig2C_p50_example_data.ipynb
        ├── 4_p50_analysis.ipynb
        ├── 5_performance_correlation.ipynb
        ├── 6_clustering_paradigms.ipynb
        ├── 7_catch_trials.ipynb
        ├── 8_linear_models.ipynb
        └── SupplFig_example_data.ipynb  - in construction
```

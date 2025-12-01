# No Time (To Die): Machine Learning for Predicting Time of Death from Gene Expression
![Poster](poster.png)



## Overview

This repository contains a comprehensive machine learning pipeline for predicting Time of Death (TOD) from circadian gene expression patterns. Gene expression levels show strong circadian (24-hour) rhythms, but many genomic datasets lack crucial timestamp information. This project addresses this gap by developing supervised ML methods to infer TOD from gene expression data alone.

**Key Innovation:** A novel two-stage dimension reduction approach using AutoEncoders to encode sequentiality in time-series gene expression data, consistently outperforming baseline methods.

## Motivation & Problem Statement

### The Problem
- Gene expression datasets often lack sample time (Time of Death / TOD) information
- Circadian genes exhibit sinusoidal expression patterns that depend on time of day
- Without timestamps, each measurement becomes an arbitrary point—temporal relationships cannot be analyzed
- Current methods ignore the sequential nature of gene expression across time periods

### The Solution
We propose a machine learning pipeline that:
1. Encodes temporal sequentiality in gene expression through AutoEncoders
2. Performs two-stage dimension reduction (SDL → PCA/other methods)
3. Trains and optimizes 16 regression models (5 single, 8 ensemble, 3 deep learning)
4. Achieves biologically significant TOD prediction accuracy


## Key Insights

### Why Sequentiality Matters for Circadian Genes

Circadian genes follow sinusoidal patterns. A single expression value (e.g., 8.1) could occur at two different times of day—once on the ascending slope and once on the descending slope. By encoding windows of consecutive values [N-1, N, N+1], the model captures the trajectory and can disambiguate which point in the cycle the current measurement represents.

### Why AutoEncoders?

1. **Interpretability:** Learned latent representations can be analyzed and understood
2. **Efficiency:** Simpler architecture than CNNs for this task
3. **Performance:** Consistently outperforms CNN baselines
4. **Flexibility:** Sliding window approach naturally encodes temporal context


## Dataset

### Source
[Chen et al. 2016](https://doi.org/10.1073/pnas.1515150113) - A study examining circadian patterns of gene expression in younger vs. older adults

### Sample Characteristics
- **Subjects:** 146 patients
- **Mean Age:** 50.7 years
- **Gender:** 75% male
- **Ethnicity:** 85% Caucasian
- **Brain Regions:** BA11 (Brodmann Area 11) and BA47 (Brodmann Area 47)
- **Total Samples:** 292 (146 patients × 2 brain areas)
- **Gene Features:** 20,000 original genes → 235 circadian genes (after feature selection)
- **Gene Expression:** 235 columns of circadian gene expression levels

### Data Format
Each observation includes:
- Time of Death (TOD) - *target variable*
- Age
- Sex (1 = Male, 0 = Female)
- Brain Area (BA11 or BA47)
- Gene Expression Values (235 circadian genes)

## Methodology

### 1. Data Preprocessing Pipeline

#### Step 1: Divide Data
- Separate datasets per brain area (BA11 and BA47)
- Process independently to account for regional differences

#### Step 2: Train/Test Split
- Test 3 different split ratios (stratified by TOD bins)
- Use binning strategy to ensure temporal distribution

#### Step 3: Normalize Data
- Test 2 normalization techniques
- Standardize gene expression values

#### Step 4: Dimension Reduction

**Stage 1: Sequentiality Encoding via AutoEncoders**
- Creates sliding windows of gene expression values
- Tests 3 window sizes (e.g., [N-1, N, N+1] for window size = 1)
- Each window is compressed to a single Encoded Value (EV)
- Encodes temporal context: previous, current, and next expression values

**Stage 2: Dimensionality Reduction**
- Tests 4 reduction methods (e.g., PCA, etc.)
- Reduces high-dimensional encoded features to lower-dimensional space
- Combats curse of dimensionality

### 2. AutoEncoder Architecture

```
Input Layer (sequence of gene values)
    ↓
Encoder (compresses window → single latent value)
    ↓
Encoded Value (single-value dimensional latent representation)
    ↓
Decoder (reconstructs original values for evaluation)
    ↓
Output Layer (reconstructed window)
```

**Key Features:**
- Learns non-linear temporal patterns
- More interpretable than CNNs
- Typically more computationally efficient than CNN alternatives
- Encodes sequentiality before training main models

### 3. Model Training & Selection

#### Regressors Tested (16 total)
- **Single Models (5):** [Specify your single models]
- **Ensemble Methods (8):** [Specify your ensemble methods]
- **Deep Learning (3):** [Specify your deep learning models]

#### Hyperparameter Optimization
- Randomized search for hyperparameter tuning
- K-fold cross-validation
- Grid/random search as appropriate per model

## Results

### Performance Metrics

All results reported as Mean Absolute Error (MAE) and Standard Deviation of Error (StdDev) in hours.

#### Baseline 1: No Encoded Sequentiality
```
BA11 - MAE: 2.424 hours, StdDev: 3.077 hours
BA47 - MAE: 3.274 hours, StdDev: 3.823 hours
```

#### Baseline 2: CNN to Encode Sequentiality
```
BA11 - MAE: 0.945 hours, StdDev: 1.107 hours
BA47 - MAE: 1.757 hours, StdDev: 2.201 hours
```

#### Our Method: AutoEncoder to Encode Sequentiality
```
BA11 - MAE: 0.839 hours, StdDev: 0.996 hours
BA47 - MAE: 1.227 hours, StdDev: 1.451 hours
```

### Key Findings

1. **Sequentiality Matters:** Including sequential information in preprocessing greatly improves model performance (~3 hours → ~0.8-1.2 hours error reduction)

2. **AutoEncoders > CNNs:**
   - Better performance metrics
   - More interpretable/explainable
   - Typically less computationally expensive
   - Cleaner architectural design for this task

3. **Biological Significance:** Prediction errors within 1-2 hours represent meaningful improvements for understanding circadian gene expression patterns

4. **Regional Consistency:** Both brain areas show consistent improvement patterns, suggesting method generalizability

## Conclusions & Future Work

### Conclusions
- Including sequentiality in input data design greatly improves model performance
- AutoEncoders perform better than CNNs, are more explainable, and typically less computationally expensive
- Our approach demonstrates biologically significant accuracy improvements

### Future Directions
- Explore larger window sizes for encoding temporal dependencies
- Thoroughly evaluate performance time across methods
- Generate gene expression profiles for out-of-sample timestamping
- Test generalization to other tissues beyond brain tissue
- Validate on additional circadian gene expression datasets
- Investigate disease-specific TOD prediction models

## Project Team

### Authors
- **Tillie Slosser** (Smith College)
- **Olivia Liau** (University of Southern California)
- **Ivan Betancourt** (Amherst College)

### Mentors/Advisors
- **Qiaochu Liu** (Worcester Polytechnic Institute)
- **Deep Suchak** (Worcester Polytechnic Institute)
- **Dr. Chun-Kit Ngan** (Worcester Polytechnic Institute)
- **Dr. Chen Fu** (UMass Chan Medical School)

### Funding
This research was supported by NSF REU Site Grant: 2349370 - *Applied Artificial Intelligence for Advanced Applications (2024-2026)*

## References

Chen, C. Y., Logan, R. W., Ma, T., Lewis, D. A., Tseng, G. C., Sibille, E., ... & Turek, F. W. (2016). Effects of aging on circadian patterns of gene expression in the human prefrontal cortex. *Proceedings of the National Academy of Sciences*, 113(1), 206-211. https://doi.org/10.1073/pnas.1515150113

Xue, X., Zong, W., Glausier, J. R., Kim, S. M., Shelton, M. A., Phan, B. N., ... & Pantazatos, S. P. (2022). Molecular rhythm alterations in prefrontal cortex and nucleus accumbens associated with opioid use disorder. *Translational Psychiatry*, 12(1), 1-13. https://doi.org/10.1038/s41398-022-01845-y



## Contact & Questions

This work is pending publication (to be published March 2026). For questions or inquiries about this research, please contact one of the authors.

---

**Citation:**
```bibtex
tba
```

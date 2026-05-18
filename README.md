# Composite-DNA-BiLSTM

**Neural Decoding for Composite DNA Storage via Bidirectional Recurrent Networks**

This repository contains the PyTorch implementation, dataset generation scripts, and evaluation pipelines for:

> **"Neural Decoding for Uniform Composite DNA Storage via Bidirectional Recurrent Networks"**
> Shubham Srivastava, Krishna Gopal Benerjee, Adrish Banerjee
> Department of Electrical Engineering, IIT Kanpur
> *Accepted at IEEE ITW 2025*

> 🔄 **An extended journal (transaction) version of this work is currently under preparation.**

---

## Overview

Composite DNA storage extends the standard four-nucleotide alphabet by encoding controlled equimolar mixtures of nucleotides at each position, enabling information densities beyond 2 bits/nucleotide. This repository proposes a **Bidirectional LSTM (Bi-LSTM)** neural decoder that processes a normalised frequency matrix aggregated from multiple noisy reads and exploits sequential dependencies across positions — outperforming all position-wise baselines (Minimum Euclidean Distance, KL Divergence, Maximum Likelihood) across all tested configurations.

---

## Alphabets

| Alphabet | Classes | Composition | Capacity (bits/pos) |
|----------|---------|-------------|---------------------|
| **A₆** | 10 | 4 pure + 6 two-nucleotide mixtures | 3.32 |
| **A₁₀** | 14 | A₆ + 4 three-nucleotide mixtures | 3.81 |
| **A₁₁** | 15 | A₁₀ + 1 four-nucleotide mixture | 3.91 |
| **A₀.₂** | 34 | Variable-ratio alphabet (η=0.2) | 5.09 |

---

## Error Profiles

### Main Paper Profiles (Illumina)

| Profile | Platform | Seq Length | Total Error Rate | Dominant Error |
|---------|----------|-----------|-----------------|----------------|
| **EZ17** | Twist + Illumina MiSeq | n=136 | ~0.29% | Substitution |
| **G15** | CustomArray + Illumina MiSeq | n=104 | ~1.21% | Deletion |
| **O17** | Twist + Illumina NextSeq | n=77 | ~0.36% | Substitution |

### Supplementary Cross-Platform Profiles

| Profile | Platform | Seq Length | Total Error Rate | Notes |
|---------|----------|-----------|-----------------|-------|
| **BOS22** | Twist + Illumina MiSeq 2022 | n=136 | ~0.067% | Ultra-low error, near-ideal |
| **R21** | Twist + Oxford Nanopore MinION | n=136 | ~4.25% | Insertion-dominated |
| **B22** | Twist + Nanopore MinION short-read | n=136 | ~3.19% | Balanced IDS errors |
| **NP22** | Twist + Nanopore Pilot Nov-2022 | n=136 | ~3.71% | Highly non-uniform per-base |
| **NPF22** | Twist + Nanopore Full Pool Nov-2022 | n=136 | ~4.05% | Highest error rate tested |

---

## Model Architecture

```
Input P ∈ ℝ^{4×n}  →  2-Layer Bi-LSTM (hidden=128, dropout=0.2)  →  Linear (256 → |A|)  →  ẑ ∈ Aⁿ
```

- **Optimizer:** AdamW (lr=1e-3, weight decay=1e-4)
- **Scheduler:** Cosine annealing with 10 warm-up epochs
- **Training:** Up to 100 epochs, early stopping (patience=10), batch size=500
- **Dataset:** 100,000 sequences per (alphabet, profile) pair; 80k train / 20k test

A separate model is trained for each (alphabet, error profile, coverage depth) combination.

---

## Results

### Performance Hierarchy

Across **all 330+ experimental configurations**, the decoder ordering is consistent:

**Bi-LSTM > KL/ML > Min.D** (Illumina profiles)

**Bi-LSTM > Min.D > KL/ML** (Nanopore profiles, where KL/ML degrades due to log-amplification of noise)

KL divergence and Maximum Likelihood decoders produce **identical decisions** in all configurations.

---

### A₆ Alphabet — Main Paper (EZ17, G15, O17)

Symbol accuracy (%) | 10 classes

| M | EZ17 Bi-LSTM | EZ17 Min.D | EZ17 KL/ML | G15 Bi-LSTM | G15 Min.D | G15 KL/ML | O17 Bi-LSTM | O17 Min.D | O17 KL/ML |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 38.36 | 38.36 | 38.36 | 38.81 | 38.81 | 38.81 | 39.05 | 39.05 | 39.05 |
| 2 | **66.60** | 65.41 | 65.41 | **67.39** | 66.43 | 66.43 | **67.77** | 67.26 | 67.26 |
| 3 | **81.90** | 77.91 | 77.91 | **82.40** | 79.45 | 79.45 | **82.81** | 80.71 | 80.71 |
| 5 | **93.91** | 76.38 | 86.63 | **94.30** | 76.73 | 88.62 | **94.58** | 77.10 | 90.39 |
| 8 | **98.16** | 83.66 | 93.54 | **98.37** | 83.63 | 94.21 | **98.48** | 83.53 | 94.78 |
| 10 | **98.84** | 93.04 | 96.33 | **98.95** | 93.30 | 97.13 | **99.00** | 93.44 | 97.76 |
| 15 | **99.80** | 97.42 | 98.69 | **99.84** | 97.56 | 99.09 | **99.85** | 97.65 | 99.31 |
| 20 | **99.95** | 97.92 | 98.65 | **99.96** | 97.90 | 99.27 | **99.96** | 97.82 | 99.62 |
| 25 | **99.98** | 99.25 | 99.44 | **99.99** | 99.25 | 99.74 | **99.99** | 99.24 | 99.89 |

---

### A₁₀ Alphabet — Main Paper (EZ17, G15, O17)

Symbol accuracy (%) | 14 classes

| M | EZ17 Bi-LSTM | EZ17 Min.D | EZ17 KL/ML | G15 Bi-LSTM | G15 Min.D | G15 KL/ML | O17 Bi-LSTM | O17 Min.D | O17 KL/ML |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 27.45 | 27.45 | 27.45 | 27.69 | 27.69 | 27.69 | 27.91 | 27.91 | 27.91 |
| 2 | **47.26** | 46.69 | 46.69 | **47.72** | 47.38 | 47.38 | **48.25** | 48.05 | 48.05 |
| 3 | **63.11** | 61.15 | 61.15 | **63.75** | 62.38 | 62.38 | **64.34** | 63.49 | 63.49 |
| 5 | **81.14** | 68.31 | 75.59 | **82.02** | 69.33 | 77.75 | **82.70** | 70.29 | 79.72 |
| 8 | **91.96** | 71.67 | 84.05 | **92.64** | 71.92 | 86.11 | **93.19** | 72.07 | 87.98 |
| 10 | **94.92** | 84.90 | 87.38 | **95.44** | 85.49 | 88.45 | **95.82** | 85.91 | 89.17 |
| 15 | **98.03** | 90.67 | 94.82 | **98.27** | 91.03 | 96.01 | **98.43** | 91.23 | 96.80 |
| 20 | **99.22** | 93.20 | 96.44 | **99.33** | 93.28 | 97.44 | **99.41** | 93.28 | 97.99 |
| 25 | **99.69** | 95.81 | 97.94 | **99.75** | 95.86 | 98.76 | **99.78** | 95.78 | 99.22 |

---

### A₁₁ Alphabet — Main Paper (EZ17, G15, O17)

Symbol accuracy (%) | 15 classes

| M | EZ17 Bi-LSTM | EZ17 Min.D | EZ17 KL/ML | G15 Bi-LSTM | G15 Min.D | G15 KL/ML | O17 Bi-LSTM | O17 Min.D | O17 KL/ML |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 25.57 | 25.57 | 25.57 | 25.91 | 25.91 | 25.91 | 26.14 | 26.14 | 26.14 |
| 2 | **44.02** | 43.57 | 43.57 | **44.61** | 44.33 | 44.33 | **45.04** | 44.87 | 44.87 |
| 3 | **58.73** | 57.11 | 57.11 | **59.44** | 58.35 | 58.35 | **60.05** | 59.36 | 59.36 |
| 5 | **76.67** | 65.14 | 71.95 | **77.60** | 66.16 | 74.05 | **78.40** | 67.11 | 75.91 |
| 8 | **88.74** | 67.57 | 81.26 | **89.64** | 67.82 | 83.50 | **90.27** | 68.00 | 85.47 |
| 10 | **92.48** | 80.68 | 84.69 | **93.22** | 81.28 | 86.04 | **93.74** | 81.69 | 87.11 |
| 15 | **96.77** | 88.82 | 92.73 | **97.17** | 89.28 | 93.93 | **97.44** | 89.59 | 94.75 |
| 20 | **98.44** | 91.19 | 95.09 | **98.67** | 91.33 | 96.26 | **98.80** | 91.36 | 97.06 |
| 25 | **99.25** | 93.57 | 96.37 | **99.37** | 93.65 | 97.60 | **99.45** | 93.64 | 98.41 |

---

### Variable-Ratio Alphabet A₀.₂ — Main Paper (EZ17, G15, O17)

Symbol accuracy (%) | 34 classes | η=0.2 | 5.09 bits/position

| M | EZ17 Bi-LSTM | EZ17 Min.D | EZ17 KL/ML | G15 Bi-LSTM | G15 Min.D | G15 KL/ML | O17 Bi-LSTM | O17 Min.D | O17 KL/ML |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 11.23 | 11.30 | 11.30 | 10.32 | 10.46 | 10.46 | 11.42 | 11.57 | 11.57 |
| 2 | **19.59** | 19.25 | 19.25 | **17.95** | 16.90 | 16.90 | **20.07** | 20.01 | 20.01 |
| 3 | **26.17** | 25.06 | 25.06 | **23.66** | 21.32 | 21.32 | **26.66** | 26.30 | 26.30 |
| 5 | **34.16** | 31.60 | 31.91 | **31.06** | 26.79 | 26.25 | **34.60** | 33.05 | 33.80 |
| 8 | **47.50** | 44.32 | 44.33 | **42.75** | 35.78 | 35.80 | **48.30** | 47.03 | 47.03 |
| 10 | **52.51** | 47.88 | 48.26 | **47.37** | 38.71 | 38.31 | **53.29** | 50.65 | 51.58 |
| 15 | **63.71** | 57.56 | 57.89 | **57.48** | 45.91 | 45.51 | **64.68** | 61.34 | 62.21 |
| 20 | **71.27** | 62.05 | 64.26 | **64.56** | 50.51 | 50.32 | **72.32** | 63.99 | 69.24 |
| 25 | **76.67** | 68.46 | 68.83 | **69.92** | 54.38 | 53.87 | **77.69** | 71.47 | 74.07 |
| 30 | **80.70** | 73.37 | 73.66 | **74.19** | 57.74 | 57.99 | **81.57** | 76.87 | 77.53 |
| 40 | **86.27** | 79.01 | 79.80 | **80.82** | 61.92 | 62.18 | **87.05** | 82.04 | 84.70 |
| 50 | **90.10** | 83.71 | 83.13 | **85.54** | 64.81 | 64.81 | **90.75** | 87.46 | 88.57 |

---

## Supplementary Results — Cross-Platform Study (5 Additional Profiles)

### A₆ Alphabet — Supplementary Profiles

Symbol accuracy (%) | 10 classes | n=136 for all

| M | BOS22 Bi-LSTM | BOS22 Min.D | BOS22 KL/ML | R21 Bi-LSTM | R21 Min.D | R21 KL/ML | B22 Bi-LSTM | B22 Min.D | B22 KL/ML | NP22 Bi-LSTM | NP22 Min.D | NP22 KL/ML | NPF22 Bi-LSTM | NPF22 Min.D | NPF22 KL/ML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 39.84 | 39.84 | 39.84 | 26.12 | 25.96 | 25.96 | 28.69 | 28.70 | 28.70 | 27.81 | 27.80 | 27.80 | 27.53 | 27.53 | 27.53 |
| 2 | **69.59** | 69.54 | 69.54 | **39.05** | 35.25 | 35.25 | **44.78** | 40.96 | 40.96 | **42.90** | 39.20 | 39.20 | **42.10** | 38.49 | 38.49 |
| 3 | **84.64** | 84.30 | 84.30 | **45.90** | 37.35 | 37.35 | **54.26** | 44.26 | 44.26 | **51.78** | 42.09 | 42.09 | **50.23** | 41.14 | 41.14 |
| 5 | **95.95** | 77.50 | 95.28 | **52.78** | 42.33 | 38.62 | **63.24** | 50.00 | 46.14 | **60.10** | 47.68 | 43.80 | **58.28** | 46.65 | 42.69 |
| 8 | **99.29** | 82.84 | 95.71 | **60.78** | 48.28 | 47.25 | **72.32** | 57.53 | 56.86 | **69.06** | 54.74 | 53.90 | **67.19** | 53.52 | 52.60 |
| 10 | **99.62** | 93.48 | 98.67 | **64.92** | 49.64 | 47.30 | **76.98** | 59.79 | 56.97 | **73.61** | 56.65 | 54.04 | **71.81** | 55.32 | 52.69 |
| 15 | **99.94** | 97.84 | 99.55 | **72.52** | 53.57 | 49.64 | **84.80** | 64.78 | 60.05 | **81.40** | 61.35 | 56.83 | **79.60** | 59.87 | 55.40 |
| 20 | **99.99** | 97.59 | 99.97 | **77.89** | 55.27 | 49.57 | **89.65** | 67.18 | 59.58 | **86.55** | 63.53 | 56.52 | **84.82** | 61.89 | 55.11 |
| 25 | **100.00** | 99.15 | 99.99 | **82.10** | 56.88 | 51.84 | **92.81** | 69.15 | 62.47 | **90.11** | 65.38 | 59.20 | **88.58** | 63.67 | 57.72 |

---

### A₁₀ Alphabet — Supplementary Profiles

Symbol accuracy (%) | 14 classes | n=136 for all

| M | BOS22 Bi-LSTM | BOS22 Min.D | BOS22 KL/ML | R21 Bi-LSTM | R21 Min.D | R21 KL/ML | B22 Bi-LSTM | B22 Min.D | B22 KL/ML | NP22 Bi-LSTM | NP22 Min.D | NP22 KL/ML | NPF22 Bi-LSTM | NPF22 Min.D | NPF22 KL/ML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 28.41 | 28.41 | 28.41 | 18.67 | 18.51 | 18.51 | 20.51 | 20.51 | 20.51 | 19.98 | 19.98 | 19.98 | 19.59 | 19.58 | 19.58 |
| 2 | **49.66** | 49.65 | 49.65 | **26.96** | 25.15 | 25.15 | **31.08** | 29.34 | 29.34 | **29.74** | 28.05 | 28.05 | **29.04** | 27.34 | 27.34 |
| 3 | **66.56** | 66.45 | 66.45 | **31.94** | 28.23 | 28.23 | **38.26** | 33.92 | 33.92 | **36.38** | 32.17 | 32.17 | **35.13** | 31.13 | 31.13 |
| 5 | **85.79** | 72.53 | 85.29 | **37.76** | 31.80 | 28.92 | **46.03** | 38.78 | 35.69 | **43.35** | 36.54 | 33.46 | **41.84** | 35.42 | 32.31 |
| 8 | **95.77** | 71.94 | 92.76 | **44.31** | 38.04 | 30.64 | **54.09** | 46.12 | 38.46 | **51.08** | 43.63 | 35.83 | **49.35** | 42.41 | 34.58 |
| 10 | **97.90** | 86.52 | 90.28 | **47.90** | 39.30 | 36.24 | **58.55** | 48.71 | 45.26 | **55.28** | 45.72 | 42.33 | **53.50** | 44.37 | 40.99 |
| 15 | **99.45** | 91.69 | 97.97 | **54.96** | 43.18 | 36.69 | **67.41** | 53.95 | 46.44 | **63.55** | 50.54 | 43.22 | **61.63** | 49.05 | 41.82 |
| 20 | **99.77** | 93.12 | 98.50 | **60.19** | 44.73 | 36.48 | **73.53** | 56.30 | 46.18 | **69.55** | 52.55 | 42.91 | **67.56** | 51.09 | 41.60 |
| 25 | **99.94** | 95.52 | 99.69 | **64.35** | 46.53 | 38.02 | **78.20** | 58.72 | 48.07 | **74.16** | 54.75 | 44.71 | **72.19** | 53.21 | 43.38 |

---

### A₁₁ Alphabet — Supplementary Profiles

Symbol accuracy (%) | 15 classes | n=136 for all

| M | BOS22 Bi-LSTM | BOS22 Min.D | BOS22 KL/ML | R21 Bi-LSTM | R21 Min.D | R21 KL/ML | B22 Bi-LSTM | B22 Min.D | B22 KL/ML | NP22 Bi-LSTM | NP22 Min.D | NP22 KL/ML | NPF22 Bi-LSTM | NPF22 Min.D | NPF22 KL/ML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 26.59 | 26.59 | 26.59 | 17.38 | 17.25 | 17.25 | 19.09 | 19.11 | 19.11 | 18.58 | 18.57 | 18.57 | 18.24 | 18.23 | 18.23 |
| 2 | **46.37** | 46.36 | 46.36 | **25.03** | 23.43 | 23.43 | **28.86** | 27.37 | 27.37 | **27.61** | 26.10 | 26.10 | **26.99** | 25.53 | 25.53 |
| 3 | **62.08** | 62.02 | 62.02 | **29.46** | 26.32 | 26.32 | **35.19** | 31.64 | 31.64 | **33.36** | 29.90 | 29.90 | **32.42** | 29.13 | 29.13 |
| 5 | **81.52** | 69.25 | 81.13 | **35.18** | 29.99 | 27.21 | **42.61** | 36.52 | 33.65 | **40.13** | 34.39 | 31.49 | **38.89** | 33.43 | 30.50 |
| 8 | **93.39** | 67.90 | 90.61 | **41.42** | 35.71 | 28.22 | **50.43** | 43.33 | 35.72 | **47.57** | 40.92 | 33.22 | **46.04** | 39.94 | 32.03 |
| 10 | **96.43** | 82.24 | 89.28 | **44.80** | 37.19 | 32.20 | **54.78** | 46.19 | 40.91 | **51.62** | 43.24 | 38.03 | **50.07** | 42.11 | 36.79 |
| 15 | **98.99** | 90.21 | 96.07 | **51.59** | 40.57 | 34.34 | **63.29** | 51.11 | 43.84 | **59.74** | 47.80 | 40.72 | **57.88** | 46.36 | 39.35 |
| 20 | **99.56** | 91.28 | 97.95 | **56.68** | 42.42 | 32.68 | **69.45** | 53.73 | 42.12 | **65.68** | 50.14 | 39.04 | **63.72** | 48.63 | 37.60 |
| 25 | **99.78** | 93.37 | 99.50 | **60.79** | 44.41 | 32.97 | **74.23** | 56.29 | 42.63 | **70.29** | 52.55 | 39.49 | **68.21** | 50.92 | 38.04 |

---

## Summary: Bi-LSTM Accuracy at M=25, A₁₁ — All 8 Profiles

| Profile | Platform | Total Error | Bi-LSTM | Best Baseline |
|---------|----------|-------------|---------|---------------|
| BOS22 | Illumina | ~0.067% | **99.78** | 99.50 |
| EZ17 | Illumina | ~0.29% | **99.25** | 96.37 |
| O17 | Illumina | ~0.36% | **99.45** | 98.41 |
| G15 | Illumina | ~1.21% | **99.37** | 97.60 |
| B22 | Nanopore | ~3.19% | **74.23** | 56.29 |
| NP22 | Nanopore | ~3.71% | **70.29** | 52.55 |
| NPF22 | Nanopore | ~4.05% | **68.21** | 50.92 |
| R21 | Nanopore | ~4.25% | **60.79** | 44.41 |

All four Illumina profiles achieve >99% Bi-LSTM accuracy at M=25. The Bi-LSTM advantage over the best baseline is **+16–18 pp** for Nanopore, versus **+1–3 pp** for Illumina — a 6–18× amplification.

---

## Bi-LSTM Advantage Over Best Baseline — A₁₁ (Supplementary)

Percentage-point improvement of Bi-LSTM over best baseline decoder:

| M | BOS22 | R21 | B22 | NP22 | NPF22 |
|---|---|---|---|---|---|
| 5 | +0.39 | +5.19 | +6.09 | +5.74 | +5.46 |
| 10 | +7.15 | +7.61 | +8.59 | +8.38 | +7.96 |
| 15 | +2.92 | +11.02 | +12.18 | +11.94 | +11.52 |
| 25 | +0.28 | +16.38 | +17.94 | +17.74 | +17.29 |

---

## Smoothing Parameter Sensitivity (KL/ML, A₆/EZ17)

| M | ε=1e-10 | ε=1e-5 | ε=1e-3 | **ε=1e-2** | ε=0.05 | ε=0.1 |
|---|---|---|---|---|---|---|
| 1 | 38.36 | 38.36 | 38.36 | **38.36** | 38.36 | 38.36 |
| 2 | 65.41 | 65.41 | 65.41 | **65.41** | 65.41 | 65.41 |
| 3 | 77.91 | 77.91 | 77.91 | **77.91** | 77.91 | 77.91 |
| 5 | **86.63** | **86.63** | **86.63** | **86.63** | 76.38 | 75.02 |
| 8 | 87.97 | 87.97 | 87.98 | **93.54** | 93.52 | 83.66 |
| 10 | 86.90 | 86.90 | 95.54 | **96.33** | 93.47 | 91.95 |
| 15 | 83.48 | 83.47 | 96.24 | **98.69** | 97.42 | 97.12 |
| 20 | 80.35 | 94.35 | 97.39 | 98.65 | **99.09** | 97.12 |
| 25 | 77.80 | 92.33 | 97.85 | 99.44 | **99.67** | 98.79 |
| **Avg** | 76.09 | 79.26 | 82.59 | **83.88** | 82.36 | 80.59 |

ε = 1e-2 used throughout all experiments. Very small values (ε ≤ 1e-5) cause severe accuracy *degradation* at high coverage due to extreme log-probability ratios.

---

## Ablation Study (A₁₁/EZ17, M=10)

| Variant | Accuracy (%) | Δ vs. Reference |
|---------|-------------|-----------------|
| Min.D (baseline) | 80.68 | −11.80 |
| KL/ML (baseline) | 84.69 | −7.79 |
| Unidirectional LSTM | 90.12 | −2.36 |
| 1-Layer Bi-LSTM | 91.16 | −1.32 |
| **2-Layer Bi-LSTM (reference)** | **92.48** | — |
| 3-Layer Bi-LSTM | 90.51 | −1.97 |
| Hidden dim. 64 | ~92.0 | ≤−0.5 |
| Hidden dim. 128 (reference) | **92.48** | — |
| Hidden dim. 256 | ~92.0 | ≤−0.5 |
| Dropout 0.0 | ~92.0 | ≤−0.5 |
| Dropout 0.4 | ~92.0 | ≤−0.5 |

**The bidirectional direction is the single most impactful architectural choice** (+2.36 pp over unidirectional). Depth and hidden dimension variations affect accuracy by at most ±0.5 pp; all variants still exceed the best baseline by at least 5.4 pp.

---

## Position-wise Analysis (A₁₁/EZ17)

Accuracy (%) and Bi-LSTM advantage by sequence region (edge vs. centre):

| M | Bi-LSTM Edge | Bi-LSTM Centre | KL/ML Edge | KL/ML Centre | Advantage Edge | Advantage Centre |
|---|---|---|---|---|---|---|
| 10 | 95.21 | 91.70 | 88.64 | 83.57 | +6.57 pp | +8.13 pp |
| 15 | 98.19 | 96.37 | 95.87 | 91.84 | +2.32 pp | +4.53 pp |
| 25 | 99.66 | 99.13 | 99.33 | 95.53 | +0.33 pp | +3.60 pp |

The Bi-LSTM's persistent centre advantage at high coverage (even as the edge gap closes) confirms that **bidirectional sequential context exploitation is the primary driver of performance gains**.

---

## Alphabet Comparison at Selected Coverage Depths (EZ17)

Bi-LSTM accuracy across alphabet types:

| M | A₆ | A₁₀ | A₁₁ | A₀.₂ |
|---|---|---|---|---|
| 5 | 93.91 | 81.14 | 76.67 | 34.16 |
| 10 | 98.84 | 94.92 | 92.48 | 52.51 |
| 15 | 99.80 | 98.03 | 96.77 | 63.71 |
| 25 | 99.98 | 99.69 | 99.25 | 76.67 |
| 50 | — | — | — | 90.10 |
| **Capacity** | **3.32** | **3.81** | **3.91** | **5.09** |

The 30% capacity increase from A₁₁ to A₀.₂ requires ~5× higher coverage for equivalent accuracy.

---

## Coverage Requirements for Target Accuracy (Main Paper)

Minimum M required to exceed target:

| Alphabet | Target | EZ17 Bi-LSTM | EZ17 KL/ML | EZ17 Min.D | G15 Bi-LSTM | G15 KL/ML | G15 Min.D | O17 Bi-LSTM | O17 KL/ML | O17 Min.D |
|---|---|---|---|---|---|---|---|---|---|---|
| A₆ | 90% | 5 | 5 | 10 | 5 | 5 | 10 | 5 | 5 | 10 |
| A₆ | 98% | 8 | 15 | 20 | 8 | 15 | 20 | 8 | 10 | 20 |
| A₁₀ | 90% | 8 | 15 | >25 | 8 | 10 | >25 | 8 | 10 | >25 |
| A₁₀ | 98% | 15 | 25 | >25 | 15 | 20 | >25 | 15 | 20 | >25 |
| A₁₁ | 90% | 8 | 15 | >25 | 8 | 15 | >25 | 8 | 10 | >25 |
| A₁₁ | 98% | 20 | >25 | >25 | 20 | >25 | >25 | 20 | 25 | >25 |

Only the Bi-LSTM achieves 98% on A₁₁ within M ≤ 25 across all profiles.

---

## Repository Structure

```
Composite-DNA-BiLSTM/
├── dataset_generator_2mix-{profile}.ipynb          # A₆ dataset generation
├── dataset_generator_2mix_3mix-{profile}.ipynb     # A₁₀ dataset generation
├── dataset_generator_2mix_3mix_4mix-{profile}.ipynb # A₁₁ dataset generation
├── dataset_generator_eta_based-{profile}.ipynb      # A₀.₂ variable-ratio generation
├── Dataset_generator_eta_cross_platform.ipynb       # Cross-platform dataset generation
├── dataset_generator_uniform_composite.py           # Standalone uniform dataset script
├── dataset_generator_eta_based.py                   # Standalone eta-based dataset script
│
├── Train_evaluate_2mix-{profile}.ipynb             # Train/eval on A₆
├── Train_evaluate_2mix_3mix-{profile}.ipynb        # Train/eval on A₁₀
├── Train_evaluate_2mix_3mix_4mix-{profile}.ipynb   # Train/eval on A₁₁
├── train_evaluate_eta_based-{profile}.ipynb        # Train/eval on A₀.₂
├── Train_evaluate_eta_cross_platform.ipynb          # Cross-platform evaluation
├── Train_evaluate_2mix_3mix_4mix-Erlich.py         # Standalone training script (Erlich/EZ17)
├── train_evaluate_eta_based-Erlich.py              # Standalone eta training script
│
├── Ablation-Train_evaluate_2mix_3mix_4mix-Erlich-{variant}.ipynb  # Ablation studies
│    ├── dropout_0.0 / dropout_0.4
│    ├── hiddendim_64 / hiddendim_256
│    ├── nlayers_1 / nlayers_3
│    └── unidirectional
│
├── Epsilon sensitivity analysis.ipynb              # Smoothing parameter sensitivity
├── Isolated-Error-Analysis.ipynb                   # Error-type isolation analysis
├── Position wise analysis.ipynb                    # Position-wise accuracy analysis
├── Position wise analysis multi coverage.ipynb     # Multi-coverage position analysis
├── Symbol_Confusion_Analysis.ipynb                 # Per-symbol confusion matrices
├── decoder-complexity-analysis.ipynb               # Decoder computational complexity
│
├── epsilon_sensitivity_results/                    # Saved epsilon sensitivity outputs
├── complexity_analysis_results/                    # Decoder complexity outputs
├── confusion_analysis_EZ17_2mix_3mix_4mix/         # Confusion matrix outputs
├── isolated_error_analysis_2mix_3mix_4mix_M10/     # Isolated error analysis outputs
├── position_analysis_results/                      # Position-wise analysis outputs
└── position_analysis_multi_coverage/               # Multi-coverage position outputs
```

`{profile}` is one of: `Erlich`, `grass`, `organick`, `R21`, `B22`, `BOS22`, `NP22`, `NPF22`

---

## Acknowledgments

We gratefully acknowledge the following resources that made this work possible:

* **[Deep-DNA-based-storage](https://github.com/itaiorr/Deep-DNA-based-storage)** by Bar-Lev et al. for the error statistics (per-nucleotide IDS rates for the R21, B22, BOS22, NP22, and NPF22 profiles) made available through their Synthetic Data Generator (SDG). These rates were characterised using the SOLQC tool and are the basis for all five supplementary cross-platform error profiles in this work.
  > D. Bar-Lev, I. Orr, O. Sabary, T. Etzion, and E. Yaakobi, "Scalable and robust DNA-based storage via coding theory and deep learning," *Nature Machine Intelligence*, pp. 1–11, 2025.

* **[SOLQC](https://github.com/omersabary/SOLQC)** — Synthetic Oligo Library Quality Control tool by Sabary et al., used to characterise the per-nucleotide error statistics of all sequencing profiles.
  > O. Sabary, Y. Orlev, R. Shafir, L. Anavy, E. Yaakobi, and Z. Yakhini, "SOLQC: Synthetic oligo library quality control tool," *Bioinformatics*, vol. 37, no. 5, pp. 720–722, 2021.

* **Erlich & Zielinski (EZ17)** for the original oligonucleotide pool (Twist Bioscience synthesis, Illumina MiSeq, n=136) which underlies both the EZ17 main-paper profile and all five supplementary profiles re-sequenced from the same pool.
  > Y. Erlich and D. Zielinski, "DNA fountain enables a robust and efficient storage architecture," *Science*, vol. 355, no. 6328, pp. 950–954, 2017.

* **Anavy et al.** for introducing the composite DNA framework and the KL divergence baseline decoder.
  > L. Anavy, I. Vaknin, O. Atar, R. Amit, and Z. Yakhini, "Data storage in DNA with fewer synthesis cycles using composite DNA letters," *Nature Biotechnology*, vol. 37, no. 10, pp. 1229–1236, 2019.

* **Cohen & Yaakobi** for the maximum likelihood decoder formulation under the multinomial observation model.
  > T. Cohen and E. Yaakobi, "Optimizing the decoding probability and coverage ratio of composite DNA," *IEEE Journal on Selected Areas in Information Theory*, vol. 6, pp. 417–431, 2025.

* **Grass et al. (G15)** and **Organick et al. (O17)** for the synthesis-sequencing error profiles used in the main paper.
  > R. N. Grass, R. Heckel, M. Puddu, D. Paunescu, and W. J. Stark, "Robust chemical preservation of digital information on DNA in silica with error-correcting codes," *Angewandte Chemie*, 2015.
  > L. Organick et al., "Random access in large-scale DNA data storage," *Nature Biotechnology*, vol. 36, pp. 242–248, 2018.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{srivastava2025compositeDNA,
  title={Neural Decoding for Uniform Composite {DNA} Storage via Bidirectional Recurrent Networks},
  author={Srivastava, Shubham and Benerjee, Krishna Gopal and Banerjee, Adrish},
  booktitle={IEEE Information Theory Workshop (ITW)},
  year={2025}
}
```

---

## Authors

**Shubham Srivastava, Krishna Gopal Benerjee, Adrish Banerjee**
Department of Electrical Engineering, Indian Institute of Technology Kanpur, India
`{shubhsr, kgopal, adrish}@iitk.ac.in`

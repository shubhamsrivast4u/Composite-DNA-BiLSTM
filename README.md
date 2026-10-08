# Composite-DNA-BiLSTM

**Neural Decoding for Composite DNA Storage via Bidirectional Recurrent Networks**

This repository contains the PyTorch implementation, dataset generation scripts, and evaluation pipelines for:

> **"Neural Decoding for Uniform Composite DNA Storage via Bidirectional Recurrent Networks"**
> Shubham Srivastava, Krishna Gopal Benerjee, Adrish Banerjee
> Department of Electrical Engineering, IIT Kanpur
> *Submitted at IEEE ITW 2026*

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
- **Dataset:** 100,000 sequences per (alphabet, profile, coverage) combination; 80k train / 20k test

A separate model is trained for each (alphabet, error profile, coverage depth) combination.

---

## Main Paper Results — EZ17, G15, O17

### A₆ Alphabet — Main Paper (EZ17, G15, O17)

Symbol accuracy (%) | 10 classes

| M | EZ17 Bi-LSTM | EZ17 Min.D | EZ17 KL/ML | G15 Bi-LSTM | G15 Min.D | G15 KL/ML | O17 Bi-LSTM | O17 Min.D | O17 KL/ML |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 37.96 | 37.96 | 37.96 | 35.10 | 35.11 | 35.11 | 39.02 | 39.02 | 39.02 |
| 3 | **80.55** | 75.67 | 75.67 | **73.13** | 64.29 | 64.29 | **82.47** | 80.46 | 80.46 |
| 5 | **92.76** | 75.58 | 83.78 | **85.17** | 68.97 | 68.97 | **94.04** | 77.06 | 89.95 |
| 10 | **98.50** | 92.48 | 94.90 | **95.12** | 85.04 | 85.11 | **98.54** | 93.46 | 97.72 |
| 15 | **99.72** | 97.13 | 97.92 | **98.42** | 91.16 | 91.17 | **99.81** | 97.63 | 99.30 |
| 25 | **99.98** | 99.23 | 98.85 | **99.76** | 95.74 | 95.20 | **99.99** | 99.24 | 99.89 |

---

### A₁₀ Alphabet — Main Paper (EZ17, G15, O17)

Symbol accuracy (%) | 14 classes

| M | EZ17 Bi-LSTM | EZ17 Min.D | EZ17 KL/ML | G15 Bi-LSTM | G15 Min.D | G15 KL/ML | O17 Bi-LSTM | O17 Min.D | O17 KL/ML |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 27.07 | 27.07 | 27.07 | 25.05 | 25.05 | 25.05 | 27.84 | 27.84 | 27.84 |
| 3 | **61.80** | 59.50 | 59.50 | **54.22** | 50.21 | 50.21 | **64.04** | 63.22 | 63.22 |
| 5 | **79.36** | 66.86 | 72.73 | **68.37** | 57.73 | 57.73 | **82.27** | 70.15 | 79.40 |
| 10 | **93.58** | 83.85 | 85.68 | **83.86** | 74.09 | 74.13 | **95.08** | 85.88 | 89.12 |
| 15 | **97.37** | 90.03 | 92.94 | **91.58** | 81.75 | 81.76 | **97.91** | 91.27 | 96.73 |
| 25 | **99.52** | 95.64 | 96.40 | **97.42** | 89.07 | 88.76 | **99.73** | 95.85 | 99.23 |

---

### A₁₁ Alphabet — Main Paper (EZ17, G15, O17)

Symbol accuracy (%) | 15 classes

| M | EZ17 Bi-LSTM | EZ17 Min.D | EZ17 KL/ML | G15 Bi-LSTM | G15 Min.D | G15 KL/ML | O17 Bi-LSTM | O17 Min.D | O17 KL/ML |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 25.28 | 25.28 | 25.28 | 23.43 | 23.43 | 23.43 | 25.92 | 25.92 | 25.92 |
| 3 | **57.56** | 55.62 | 55.62 | **50.19** | 46.88 | 46.88 | **59.64** | 59.02 | 59.02 |
| 5 | **74.72** | 63.57 | 69.00 | **63.65** | 54.57 | 54.57 | **77.84** | 66.82 | 75.47 |
| 10 | **90.73** | 79.55 | 82.40 | **79.84** | 70.18 | 70.21 | **92.90** | 81.67 | 86.99 |
| 15 | **95.73** | 87.92 | 90.72 | **88.19** | 78.69 | 78.70 | **96.77** | 89.58 | 94.72 |
| 25 | **98.92** | 93.34 | 94.26 | **95.72** | 86.51 | 86.30 | **99.36** | 93.69 | 98.37 |

---

### Variable-Ratio Alphabet A₀.₂ — Main Paper (EZ17, G15, O17)

Symbol accuracy (%) | 34 classes | η=0.2 | 5.09 bits/position

| M | EZ17 Bi-LSTM | EZ17 Min.D | EZ17 KL/ML | G15 Bi-LSTM | G15 Min.D | G15 KL/ML | O17 Bi-LSTM | O17 Min.D | O17 KL/ML |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 11.16 | 11.21 | 11.21 | 10.23 | 10.31 | 10.31 | 11.24 | 11.43 | 11.43 |
| 3 | **25.69** | 24.41 | 24.41 | **23.08** | 20.72 | 20.72 | **26.34** | 25.84 | 25.84 |
| 5 | **33.56** | 30.74 | 30.80 | **30.31** | 26.07 | 25.53 | **34.29** | 32.60 | 33.25 |
| 10 | **51.52** | 46.26 | 46.46 | **46.10** | 37.42 | 36.95 | **52.56** | 49.74 | 50.49 |
| 15 | **62.43** | 55.52 | 55.66 | **55.91** | 44.16 | 42.61 | **63.93** | 60.15 | 60.85 |
| 25 | **75.23** | 66.29 | 66.19 | **68.31** | 52.46 | 52.99 | **76.69** | 70.75 | 72.24 |
| 50 | **89.24** | 80.80 | 80.19 | **84.28** | 62.42 | 63.32 | **90.20** | 86.73 | 87.13 |

---

## Supplementary Results — Cross-Platform Study (5 Additional Profiles)

### A₆ Alphabet — Supplementary Profiles

Symbol accuracy (%) | 10 classes | n=136 for all

| M | BOS22 Bi-LSTM | BOS22 Min.D | BOS22 KL/ML | R21 Bi-LSTM | R21 Min.D | R21 KL/ML | B22 Bi-LSTM | B22 Min.D | B22 KL/ML | NP22 Bi-LSTM | NP22 Min.D | NP22 KL/ML | NPF22 Bi-LSTM | NPF22 Min.D | NPF22 KL/ML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 39.84 | 39.84 | 39.84 | 24.53 | 24.13 | 24.13 | 27.00 | 26.92 | 26.92 | 25.96 | 25.78 | 25.78 | 25.72 | 25.51 | 25.51 |
| 3 | **84.67** | 84.49 | 84.49 | **40.52** | 33.43 | 33.43 | **48.74** | 39.95 | 39.95 | **45.04** | 37.08 | 37.08 | **44.24** | 36.62 | 36.62 |
| 5 | **95.95** | 77.49 | 95.55 | **45.77** | 37.68 | 38.60 | **56.29** | 45.42 | 45.88 | **51.53** | 41.91 | 42.66 | **50.62** | 41.40 | 42.17 |
| 10 | **99.60** | 93.45 | 98.70 | **56.21** | 44.18 | 46.19 | **68.72** | 53.79 | 55.76 | **63.44** | 49.74 | 51.71 | **62.31** | 48.94 | 50.98 |
| 15 | **99.93** | 97.85 | 99.92 | **62.50** | 47.65 | 49.56 | **76.19** | 58.21 | 60.16 | **70.52** | 53.59 | 55.57 | **69.47** | 53.05 | 55.03 |
| 25 | **100.00** | 99.14 | 100.00 | **70.74** | 50.45 | 53.09 | **84.98** | 61.86 | 64.82 | **79.30** | 56.97 | 59.81 | **78.41** | 56.26 | 59.16 |

---

### A₁₀ Alphabet — Supplementary Profiles

Symbol accuracy (%) | 14 classes | n=136 for all

| M | BOS22 Bi-LSTM | BOS22 Min.D | BOS22 KL/ML | R21 Bi-LSTM | R21 Min.D | R21 KL/ML | B22 Bi-LSTM | B22 Min.D | B22 KL/ML | NP22 Bi-LSTM | NP22 Min.D | NP22 KL/ML | NPF22 Bi-LSTM | NPF22 Min.D | NPF22 KL/ML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 28.45 | 28.45 | 28.45 | 17.51 | 17.22 | 17.22 | 19.24 | 19.20 | 19.20 | 18.49 | 18.34 | 18.34 | 18.39 | 18.26 | 18.26 |
| 3 | **66.69** | 66.64 | 66.64 | **28.46** | 25.18 | 25.18 | **34.19** | 30.41 | 30.41 | **31.56** | 28.07 | 28.07 | **31.27** | 27.83 | 27.83 |
| 5 | **85.81** | 72.67 | 85.52 | **33.38** | 28.05 | 28.62 | **40.78** | 34.48 | 34.37 | **37.39** | 31.56 | 31.83 | **36.88** | 31.22 | 31.54 |
| 10 | **97.89** | 86.54 | 96.78 | **41.92** | 34.36 | 36.03 | **51.80** | 43.11 | 44.65 | **47.45** | 39.29 | 40.96 | **46.84** | 38.84 | 40.53 |
| 15 | **99.42** | 91.71 | 98.58 | **47.78** | 37.63 | 39.85 | **59.40** | 47.58 | 49.76 | **54.36** | 43.31 | 45.54 | **53.59** | 42.72 | 44.98 |
| 25 | **99.93** | 95.49 | 99.91 | **55.35** | 40.38 | 44.01 | **69.08** | 51.43 | 55.28 | **63.09** | 46.47 | 50.25 | **62.35** | 46.00 | 49.81 |

---

### A₁₁ Alphabet — Supplementary Profiles

Symbol accuracy (%) | 15 classes | n=136 for all

| M | BOS22 Bi-LSTM | BOS22 Min.D | BOS22 KL/ML | R21 Bi-LSTM | R21 Min.D | R21 KL/ML | B22 Bi-LSTM | B22 Min.D | B22 KL/ML | NP22 Bi-LSTM | NP22 Min.D | NP22 KL/ML | NPF22 Bi-LSTM | NPF22 Min.D | NPF22 KL/ML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 26.58 | 26.58 | 26.58 | 16.35 | 16.08 | 16.08 | 18.01 | 17.96 | 17.96 | 17.29 | 17.17 | 17.17 | 17.16 | 17.02 | 17.02 |
| 3 | **62.17** | 62.14 | 62.14 | **26.46** | 23.52 | 23.52 | **31.61** | 28.39 | 28.39 | **29.28** | 26.28 | 26.28 | **28.93** | 25.99 | 25.99 |
| 5 | **81.57** | 69.30 | 81.35 | **31.20** | 26.37 | 27.00 | **37.86** | 32.51 | 32.49 | **34.89** | 29.83 | 30.13 | **34.39** | 29.48 | 29.86 |
| 10 | **96.41** | 82.24 | 95.35 | **39.52** | 32.55 | 34.25 | **48.57** | 40.77 | 42.35 | **44.50** | 37.20 | 38.82 | **43.92** | 36.69 | 38.37 |
| 15 | **98.96** | 90.20 | 97.95 | **45.10** | 35.24 | 37.91 | **55.90** | 44.80 | 47.26 | **51.11** | 40.61 | 43.24 | **50.42** | 40.00 | 42.66 |
| 25 | **99.76** | 93.33 | 99.72 | **52.40** | 38.27 | 41.88 | **65.61** | 49.20 | 53.05 | **59.91** | 44.42 | 48.21 | **59.13** | 43.82 | 47.71 |

---

### Variable-Ratio Alphabet A₀.₂ — Supplementary Profiles

Symbol accuracy (%) | 34 classes | η=0.2 | 5.09 bits/position | n=136 for all

| M | BOS22 Bi-LSTM | BOS22 Min.D | BOS22 KL/ML | R21 Bi-LSTM | R21 Min.D | R21 KL/ML | B22 Bi-LSTM | B22 Min.D | B22 KL/ML | NP22 Bi-LSTM | NP22 Min.D | NP22 KL/ML | NPF22 Bi-LSTM | NPF22 Min.D | NPF22 KL/ML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 11.69 | 11.71 | 11.71 | 7.16 | 7.12 | 7.12 | 7.83 | 7.91 | 7.91 | 7.58 | 7.57 | 7.57 | 7.53 | 7.52 | 7.52 |
| 3 | **27.19** | 27.14 | 27.14 | **12.38** | 10.67 | 10.67 | **15.02** | 12.78 | 12.78 | **13.87** | 11.89 | 11.89 | **13.70** | 11.75 | 11.75 |
| 5 | **35.23** | 34.07 | 35.22 | **15.09** | 12.67 | 12.03 | **18.90** | 15.44 | 14.70 | **17.15** | 14.25 | 13.54 | **16.94** | 14.09 | 13.37 |
| 10 | **54.35** | 52.73 | 54.16 | **21.24** | 15.95 | 15.43 | **27.62** | 20.22 | 19.34 | **24.71** | 18.35 | 17.65 | **24.27** | 18.03 | 17.41 |
| 15 | **66.23** | 64.39 | 65.91 | **25.83** | 18.01 | 17.37 | **33.87** | 23.06 | 21.91 | **30.33** | 20.83 | 19.93 | **29.70** | 20.41 | 19.62 |
| 25 | **79.54** | 71.69 | 79.04 | **32.96** | 20.31 | 19.43 | **43.34** | 26.34 | 24.78 | **38.77** | 23.63 | 22.42 | **38.08** | 23.30 | 22.21 |
| 50 | **91.87** | 87.84 | 91.31 | **44.51** | 23.00 | 21.87 | **58.04** | 30.14 | 28.22 | **52.31** | 26.91 | 25.40 | **51.43** | 26.51 | 25.08 |

---

## Summary: Bi-LSTM Accuracy at M=25, A₁₁ — All 8 Profiles

| Profile | Platform | Total Error | Bi-LSTM | Best Baseline | Advantage |
|---------|----------|-------------|---------|---------------|-----------|
| BOS22 | Illumina | ~0.067% | **99.76** | 99.72 (KL) | +0.04 pp |
| O17 | Illumina | ~0.36% | **99.36** | 98.37 (KL) | +0.99 pp |
| EZ17 | Illumina | ~0.29% | **98.92** | 94.26 (KL) | +4.66 pp |
| G15 | Illumina | ~1.21% | **95.72** | 86.51 (Min.D) | +9.21 pp |
| B22 | Nanopore | ~3.19% | **65.61** | 53.05 (KL) | +12.56 pp |
| NP22 | Nanopore | ~3.71% | **59.91** | 48.21 (KL) | +11.70 pp |
| NPF22 | Nanopore | ~4.05% | **59.13** | 47.71 (KL) | +11.42 pp |
| R21 | Nanopore | ~4.25% | **52.40** | 41.88 (KL) | +10.52 pp |

For Illumina profiles, the Bi-LSTM advantage ranges from **+0.04 pp** (BOS22, near-ideal) to **+9.21 pp** (G15, deletion-dominated). For Nanopore profiles the advantage is consistently **+10.5–12.6 pp**.

---

## Bi-LSTM Advantage Over Best Baseline — A₁₁ (Supplementary)

Percentage-point improvement of Bi-LSTM over best baseline decoder:

| M | BOS22 | R21 | B22 | NP22 | NPF22 |
|---|---|---|---|---|---|
| 5 | +0.23 | +4.20 | +5.36 | +4.76 | +4.53 |
| 10 | +1.07 | +5.28 | +6.22 | +5.68 | +5.54 |
| 15 | +1.00 | +7.19 | +8.64 | +7.87 | +7.76 |
| 25 | +0.05 | +10.52 | +12.56 | +11.70 | +11.42 |

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

ε = 1e-2 used throughout all experiments. Very small values (ε ≤ 1e-5) cause severe accuracy degradation at high coverage due to extreme log-probability ratios.

---

## Ablation Study (A₁₁/EZ17, M=10)

| Variant | Accuracy (%) | Δ vs. Reference |
|---------|-------------|-----------------|
| Min.D (baseline) | 79.55 | −11.18 |
| KL/ML (baseline) | 82.40 | −8.33 |
| Unidirectional LSTM | 88.25 | −2.48 |
| 1-Layer Bi-LSTM | 89.07 | −1.66 |
| **2-Layer Bi-LSTM (reference)** | **90.73** | — |
| 3-Layer Bi-LSTM | 90.96 | +0.22 |
| Hidden dim. 64 | 90.23 | −0.51 |
| Hidden dim. 128 (reference) | **90.73** | — |
| Hidden dim. 256 | 90.82 | +0.09 |
| Dropout 0.1 | 90.67 | −0.06 |
| Dropout 0.3 | 90.71 | −0.03 |

**The bidirectional direction is the single most impactful architectural choice** (−2.48 pp vs. unidirectional). Depth and hidden dimension variations affect accuracy by at most ±0.5 pp; all variants still exceed the best baseline (KL/ML) by at least 6.6 pp.

---

## Position-wise Analysis (A₁₁/EZ17)

Accuracy (%) and Bi-LSTM advantage by sequence region (edge vs. centre):

| M | Bi-LSTM Edge | Bi-LSTM Centre | KL/ML Edge | KL/ML Centre | Advantage Edge | Advantage Centre |
|---|---|---|---|---|---|---|
| 5 | 79.82 | 73.27 | 78.75 | 66.23 | +1.06 pp | +7.04 pp |
| 10 | 94.81 | 89.58 | 88.42 | 80.70 | +6.39 pp | +8.88 pp |
| 15 | 97.99 | 95.09 | 95.78 | 89.28 | +2.21 pp | +5.81 pp |
| 25 | 99.64 | 98.72 | 99.28 | 92.84 | +0.36 pp | +5.88 pp |

The Bi-LSTM's persistent centre advantage at high coverage (even as the edge gap closes) confirms that **bidirectional sequential context exploitation is the primary driver of performance gains** at positions far from sequence boundaries.

---

## Alphabet Comparison at Selected Coverage Depths (EZ17)

Bi-LSTM accuracy across alphabet types:

| M | A₆ | A₁₀ | A₁₁ | A₀.₂ |
|---|---|---|---|---|
| 5 | 92.76 | 79.36 | 74.72 | 33.56 |
| 10 | 98.50 | 93.58 | 90.73 | 51.52 |
| 15 | 99.72 | 97.37 | 95.73 | 62.43 |
| 25 | 99.98 | 99.52 | 98.92 | 75.23 |
| 50 | — | — | — | 89.24 |
| **Capacity** | **3.32** | **3.81** | **3.91** | **5.09** |

The 30% capacity increase from A₁₁ to A₀.₂ requires approximately 5× higher coverage for equivalent accuracy.

---

## Coverage Requirements for Target Accuracy

Minimum M required to exceed target accuracy (Bi-LSTM / best baseline):

| Alphabet | Target | EZ17 | G15 | O17 |
|---|---|---|---|---|
| A₆ | 90% | **5** / 10 | **5–10** / >10 | **5** / 10 |
| A₆ | 98% | **10** / 25 | **10–15** / >25 | **10** / 10 |
| A₁₀ | 90% | **10** / 15 | **15–25** / >25 | **10** / 15 |
| A₁₀ | 98% | **≤25** / >25 | **>25** / >25 | **≤25** / >25 |
| A₁₁ | 90% | **10** / >25 | **>25** / >25 | **10** / >25 |
| A₁₁ | 98% | **25** / >25 | **>25** / >25 | **≤25** / >25 |

Only the Bi-LSTM achieves 98% on A₁₁ within M ≤ 25 for EZ17 and O17. G15's deletion-dominated error profile requires higher coverage due to the difficulty of frequency estimation under insertions/deletions.

---

## Repository Structure

```
Composite-DNA-BiLSTM/
├── .gitignore
│
├── dataset_generator_uniform_composite.py           # Standalone uniform dataset script
├── dataset_generator_eta_based.py                   # Standalone eta-based dataset script
│
├── train_evaluate_uniform_composite-"2mix_only"-{profiles}.ipynb    # Train/eval on A₆
├── train_evaluate_uniform_composite-"2mix_3mix"-{profiles}.ipynb    # Train/eval on A₁₀
├── train_evaluate_uniform_composite-"2mix_3mix_4mix"-{profiles}.ipynb # Train/eval on A₁₁
├── train_evaluate_eta_based-{profiles}.ipynb        # Train/eval on A₀.₂ (paired profiles)
├── train_evaluate_eta_based_3mix.ipynb              # 3-mix eta-based training
├── train_evaluate_eta_based_4mix.ipynb              # 4-mix eta-based training
│
├── Ablation-train_evaluate_uniform_composite-"2mix_3mix_4mix"-"EZ17"-{variant}.ipynb
│    ├── Unidirectional
│    ├── dropout0.1 / dropout0.3
│    ├── hidden64 / hidden256
│    └── nlayers1 / nlayers3
│
├── Isolated_Error_Analysis.ipynb                    # Error-type isolation analysis
├── Position_wise_analysis_multi_coverage.ipynb      # Position-wise accuracy (multi-coverage)
├── verify_theory_on_real_data.py                    # Theory verification script
│
├── Results/
│    ├── {PROFILE}_{ALPHABET}_{HPARAMS}/             # Per-experiment results
│    │    ├── experiment_results.txt                 # Full accuracy/NER/SER report
│    │    ├── experiment_results.mat                 # MATLAB-format results
│    │    ├── all_training_histories.json            # All coverage training curves
│    │    ├── training_history_M{N}.json             # Per-coverage training history
│    │    ├── final_comparison_plot.png              # Summary comparison figure
│    │    ├── best_model_M{N}.pth                   # Best checkpoint per coverage
│    │    └── final_model_M{N}.pth                  # Final model per coverage
│    │
│    └── Ablation_study_{variant}/                  # Ablation experiment results
│
└── position_analysis_multi_coverage/               # Position-wise analysis outputs
     └── EZ17_2mix_3mix_4mix_{HPARAMS}/
          └── multi_coverage_results.json
```

`{profiles}` = paired groups, e.g. `"EZ17", "G15", "O17", "R21"` or `"B22", "BOS22", "NP22", "NPF22"`

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

# Latency-Aware Machine Learning-Guided Feature Selection for Multi-Class Motor Imagery EEG Decoding

*This repository contains the official codebase for the paper submitted to the 2026 IEEE Engineering in Medicine and Biology Society (EMBC) conference.*

## 📄 Abstract
Brain-computer interfaces that use EEG can be susceptible to high dimensional feature spaces that can contain redundant or weakly informative features, poor generalization, and growing latency. Our hypothesis is a leakage-safe, machine-learned feature selection framework combining multi-domain handcrafted EEG features with common spatial pattern (CSP) features and selecting small subsets with nested cross-validation. 

Evaluated on BCI Competition IV Dataset 2a (T session, 9 subjects, 4-class motor imagery), the proposed method achieved **61.4 ± 18.0% accuracy, 0.612 ± 0.181 macro-F1, and 0.486 ± 0.240 Cohen's κ**, outperforming a full-feature baseline (53.5 ± 15.9%, 0.531 ± 0.160, 0.380 ± 0.212). 

Classification inference latency decreased from 0.046 ± 0.007 ms/trial to **0.030 ± 0.011 ms/trial (~35% reduction)**. Across outer folds, the selected subset size was most frequently k = 20 (82.2% of folds), supporting low-latency deployment. These results highlight a favorable performance-complexity trade-off for multi-class MI EEG decoding.

[Figure_Baseline_Tonly_PerSubject_MacroF1](Figure_Baseline_Tonly_PerSubject_MacroF1.png)

[Figure_Tonly_PerSubject_MacroF1](Figure_Tonly_PerSubject_MacroF1.png)

---

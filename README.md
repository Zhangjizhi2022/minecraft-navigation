# Minecraft Navigation: 3D Path Analysis

This repository contains the complete codebase for analyzing participant behavior in the **Minecraft Memory and Navigation Task (MNN)**, with a focus on spatial navigation, learning efficiency, and cognitive map development in 3D environments.

Associated manuscript:

> **Navigating Cognitive Maps: Statistical Analysis of 3D Path Data in Minecraft**  
> *Psychometrika, 2025 (Revise & Resubmit)*

---

## 📁 Directory Structure

```
.
├── data/           # Input and intermediate data files (by environment: 1E–4E)
├── python/         # Python scripts for preprocessing, Dijkstra, clustering, and plotting
├── R/              # R scripts for functional data analysis and regression
└── README.md       # Project overview and usage instructions
```

---

## 🚀 Getting Started

### Python

1. Install required packages:
```bash
pip install numpy pandas networkx matplotlib
```

2. Run scripts:
```bash
python python/Dijkstra.py
python python/object_to_object_1.py
python python/segment_log.py
python python/test.py
```

### R

1. Install required packages:
```r
install.packages(c("fda", "splines", "ggplot2", "cluster"))
```

2. Run analysis:
```r
source("R/fregression_final.R")
```

---

## 📊 Data Format

Each folder under `data/` (e.g., `1E/`, `2E/`) contains:

- 3D path logs of training and test sessions
- Optimal path cost matrices (Dijkstra)
- Segment-level cost difference curves
- Cluster assignments and aggregated summaries

---

## 🧠 Key Analyses

| Step                     | Script                        | Description                                        |
|--------------------------|-------------------------------|----------------------------------------------------|
| Terrain Graph Modeling   | `Dijkstra.py`                 | Builds weighted graph of the 3D world using block topology |
| Cost Curve Construction  | `object_to_object_1.py`       | Converts navigation into normalized cost functions |
| Segment Clustering       | `segment_log.py`              | Performs functional clustering of path segments    |
| Test Phase Evaluation    | `test.py`                     | Measures accuracy and efficiency of placement      |
| Functional Regression    | `fregression_final.R`         | Relates training behavior to test outcomes         |

---

## 📜 License

This project is licensed under the MIT License.

---

## 📚 Citation

```bibtex
@article{zhang2025minecraft,
  title={Navigating Cognitive Maps: Statistical Analysis of 3D Path Data in Minecraft},
  author={Zhang, Jizhi and others},
  journal={Psychometrika},
  year={2025}
}
```

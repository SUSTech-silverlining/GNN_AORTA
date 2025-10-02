# A Framework for GNN-Based Aortic Analysis

### 🚀 Project Overview

Welcome to my project portfolio. This repository documents my core contributions to a research project focused on **Graph Neural Networks (GNNs) for Aortic Analysis**.

In this project, I was responsible for adapting and extending a proprietary in-house hybrid 1D-3D GNN. A significant part of my work also involved implementing, and rigorously evaluating several baseline models from state-of-the-art literature.

The primary purpose of this repository is to **showcase my technical contributions, problem-solving methodologies, and code implementation skills**.

**⚠️ A Note on the Current Status:**

  * **Proprietary Model:** For confidentiality reasons, the primary in-house model currently under development is not included in this repository.
  * **Ongoing Research:** As this is an active research project, the repository's structure and code are subject to ongoing refinement. The current state is intended to best reflect my individual contributions.

-----

### 🗺️ Repository Structure

This repository is organized as follows to provide a clear roadmap of the project:

```
.
├── 📂 g_unet/               # Baseline Model 1: Multi-scale Graph U-Net
│   └── main.ipynb
├── 📂 lab-gatr/             # Baseline Model 2: GNN with a Transformer
│   └── main.ipynb
├── 📂 single_scale/         # Baseline Model 3: Single-Scale GNN (MGN)
│   └── main.ipynb
├── 📂 load_suk_data/        # Custom scripts for processing a specific external dataset
└── 📂 src/                  # Core source code
    ├── dataset.py
    ├── dataset_trans.py
    ├── train.py
    └── train_trans.py
```

  * **/g\_unet, /lab-gatr, /single\_scale**: Each directory corresponds to one of the three baseline models I implemented. The `main.ipynb` file in each contains a complete experimental pipeline, from data loading and model training to evaluation.
  * **/load\_suk\_data/**: During our research, we needed to integrate a public dataset that was incompatible with our existing pipelines. This folder contains the scripts I **independently authored to parse, clean, and transform this dataset**, resolving critical compatibility issues.
  * **/src/**: This directory contains the project's core modules.
      * `dataset.py` & `train.py`: These are the general-purpose data loading and training modules used for the `g_unet` and `single_scale` models. I **heavily refactored** these from existing laboratory code to improve modularity and flexibility.
      * `dataset_trans.py` & `train_trans.py`: These are **custom-tailored versions** built specifically for the `lab-gatr` model to handle its unique data transformation requirements.

-----

### ✨ My Core Contributions

My work on this project can be broken down into three main areas: model implementation and pipeline engineering, data pipeline design, and an in-depth exploration of normalization strategies.

#### 🧠 1. Model Implementation & Pipeline Engineering

  * **End-to-End Pipeline Construction**: I built the experimental pipelines from scratch in the `main.ipynb` notebooks for all three baseline models (`g_unet`, `lab-gatr`, and `single_scale`), creating a highly unified and modular framework.
  * **Code Unification**: Despite the architectural differences between the models, I architected the pipelines to maximize code reuse. The primary distinctions between them were elegantly managed through configurable **data transforms** and **input features**.
  * **Model Provenance**:
      * **Multi-scale Graph U-Net (g\_unet)**: Inspired by the work at [sukjulian/coronary-mesh-convolution](https://www.google.com/search?q=https://github.com/sukjulian/coronary-mesh-convolution.git).
      * **Lab-GATR**: A GNN incorporating Transformer concepts, adapted from [sukjulian/lab-gatr](https://www.google.com/search?q=https://github.com/sukjulian/lab-gatr.git).
      * **Single-Scale GNN (MGN)**: The single-scale graph neural network model I implemented.

#### 🔧 2. Data Preprocessing & Pipeline Design

  * **Core Module Refactoring**: I re-architected the core data loading and training modules in the `src` directory to flexibly accommodate the diverse requirements of different models and datasets.
  * **Solving Data Incompatibility**: I independently developed the complete data wrangling solution in `/load_suk_data/` to resolve format mismatches between an external dataset and our internal data structures, clearing a major obstacle for our experiments.
  * **Transform Strategy Development**: I experimented with and implemented a variety of data transformation techniques to engineer more effective input features for the models.

#### 📊 3. Normalization Strategy & Tuning

  * **Focused Research on Normalization**: A **major focus of my work** on this project was to investigate and fine-tune the **normalization methods** within the models.
  * **Systematic Experimentation**: To improve training stability and performance, I designed and executed a series of systematic experiments. This core task involved evaluating various normalization layers (e.g., `LayerNorm`, `BatchNorm`) at different points within the model architectures. Through this rigorous process, I identified the optimal normalization strategy for our specific graph-structured data.
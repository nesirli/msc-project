# Interpretable Deep-Learning and Ensemble Models for Predicting Multidrug Resistance in Klebsiella pneumoniae

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Snakemake](https://img.shields.io/badge/Snakemake-7.0+-blue.svg)](https://snakemake.readthedocs.io/)

A comprehensive, reproducible pipeline for predicting antimicrobial resistance in *Klebsiella pneumoniae* using interpretable machine learning and deep learning models.

## 🎯 Overview

This project develops an interpretable, genome-based pipeline that predicts phenotypic resistance to four key antibiotics:
- **Amikacin** (aminoglycoside)
- **Ciprofloxacin** (fluoroquinolone)  
- **Ceftazidime** (3rd-generation cephalosporin)
- **Meropenem** (carbapenem)

### Key Features

- **5 Model Types**: XGBoost, LightGBM, 1D-CNN, Sequence CNN, DNABERT-2
- **Interpretable Results**: Consensus feature importance + motif-level analysis
- **Geographic-Temporal CV**: Prevents data leakage using location-year grouping
- **Standardized Evaluation**: Consistent metrics with statistical testing
- **Reproducible Workflow**: Snakemake pipeline with 19 organized steps

## 🚀 Quick Start

### Prerequisites

#### Option 1: Docker (Recommended)
- **Docker** (20.10+) and **Docker Compose** (v2.0+)
- **8+ GB RAM** (16+ GB recommended for deep learning)
- **4+ CPU cores** (8+ recommended)

#### Option 2: Google Cloud Platform
- **GCP Account** with billing enabled
- **gcloud CLI** installed and authenticated
- **Docker** (for local building)

#### Option 3: Native Installation
- **Miniconda/Anaconda** (for environment management)
- **8+ GB RAM** (16+ GB recommended for deep learning)
- **Mac M3 Pro or equivalent** (original development environment)

### Installation

#### 🐳 Docker Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/your-username/amr-kpneumoniae-prediction.git
cd amr-kpneumoniae-prediction

# Quick start with Make
make build          # Build container
make test           # Test setup
make run            # Run pipeline

# Or use scripts directly
./scripts/docker/build.sh
./scripts/docker/run.sh --dry-run
```

#### ☁️ Google Cloud Platform

```bash
# Deploy to Cloud Run (serverless)
./scripts/gcp/deploy-cloudrun.sh --project YOUR_PROJECT_ID --allow-unauthenticated

# Or setup GKE cluster
./scripts/gcp/setup-gke.sh --project YOUR_PROJECT_ID

# Use Cloud Build for CI/CD
gcloud builds submit --config scripts/gcp/cloudbuild.yaml
```

#### 🏠 Native Installation

```bash
# Clone repository
git clone https://github.com/your-username/amr-kpneumoniae-prediction.git
cd amr-kpneumoniae-prediction

# Install Snakemake
conda install -c bioconda -c conda-forge snakemake>=7.0.0

# Download reference data (optional - included in pipeline)
# Data will be automatically downloaded during first run
```

### Basic Usage

#### Docker Commands

```bash
# Run full pipeline
make run

# Interactive shell
make shell

# Specific stages
make tree-models     # Tree models only
make dl-models       # Deep learning only
make interpret       # Interpretability analysis

# Custom parameters
./scripts/docker/run.sh --cores 8 --memory 16G --target preprocess
```

#### Google Cloud Commands

```bash
# Cloud Run deployment
curl "https://your-service-url" -H "Content-Type: application/json" \
     -d '{"target": "tree_models", "cores": 4}'

# GKE execution
kubectl exec -it deployment/amr-pipeline -n amr-pipeline -- \
        snakemake --cores 4 tree_models
```

#### Native Commands

```bash
# Run full pipeline (all 19 steps)
snakemake --use-conda --cores 8

# Run specific stages
snakemake --use-conda --cores 8 preprocess        # Steps 1-5: Data preprocessing
snakemake --use-conda --cores 8 tree_models       # Steps 14-15: Tree models only
snakemake --use-conda --cores 8 dl_models         # Steps 16-18: Deep learning only
snakemake --use-conda --cores 8 interpretability  # Step 19: Analysis only

# Run for specific antibiotic
snakemake --use-conda --cores 8 results/models/xgboost/amikacin_results.json
```

## 📊 Pipeline Architecture

### 19-Stage Workflow

| Stage | Description | Output |
|-------|-------------|--------|
| **1-5** | **Data Preprocessing** | QC, assembly, contamination screening |
| **6-8** | **Feature Extraction** | AMR genes, SNPs, k-mers |
| **9-10** | **Feature Engineering** | Selection, batch correction |
| **11-13** | **Data Preparation** | Train/test splits, k-mer/DNABERT datasets |
| **14-15** | **Tree Models** | XGBoost, LightGBM with nested CV |
| **16-18** | **Deep Learning** | 1D-CNN, Sequence CNN, DNABERT-2 |
| **19** | **Interpretability** | Cross-model analysis, motifs, statistics |

### Model Specifications

| Model | Input Features | Architecture | Key Parameters |
|-------|---------------|--------------|----------------|
| **XGBoost** | AMR genes + SNPs (179 features) | Gradient boosting | Geographic-temporal CV, class weights |
| **LightGBM** | AMR genes + SNPs (179 features) | Gradient boosting | Consistent class balancing |
| **1D-CNN** | K-mer spectra (11-mers) | Conv1D + Dense | Class-weighted loss |
| **Sequence CNN** | Raw DNA sequences | Conv1D on one-hot | Geographic-temporal grouping |
| **DNABERT-2** | 250bp read tokens | Transformer (117M params) | Fine-tuned from HuggingFace |

### Cross-Validation Strategy

- **Geographic-Temporal Grouping**: Samples from same location-year stay together
- **Prevents Data Leakage**: Accounts for epidemiological structure
- **5-Fold CV**: Consistent across all models
- **Fallback**: Stratified CV when insufficient groups

### Class Imbalance Handling

- **Tree Models**: `scale_pos_weight` (neg_count/pos_count)
- **Deep Models**: `class_weight` in loss function
- **Consistent Ratios**: Shared utilities ensure fair model comparison

## 📈 Results and Evaluation

### Success Criteria
- **Primary**: F1 score ≥ 0.85 per antibiotic
- **Secondary**: Balanced accuracy ≥ 0.85
- **Analysis**: Discrepancy analysis if balanced accuracy meets threshold but F1 doesn't

### Statistical Testing
- **DeLong's Test**: ROC curve comparison between models
- **Bootstrap CI**: Confidence intervals for performance metrics
- **Bonferroni Correction**: Multiple comparison adjustment
- **Friedman Test**: Non-parametric model comparison

### Interpretability Analysis
- **Consensus Features**: Important features across multiple models
- **Motif Analysis**: Sequence patterns from CNN filters and DNABERT attention
- **Cross-Model Agreement**: Jaccard similarity of important features
- **Clinical Relevance**: Alignment with known resistance mechanisms

## 🔧 Configuration

### Key Configuration Files

```yaml
# config/config.yaml
antibiotics: [amikacin, ciprofloxacin, ceftazidime, meropenem]

splits:
  train_cutoff: 2022          # Pre-2023 for training
  test_years: [2023, 2024]    # 2023-24 for temporal validation

models:
  cv_folds: 5
  random_state: 42
  
feature_selection:
  method: "chi2_mi"
  n_features: 500
  alpha: 0.05
```

### Environment Management

The pipeline uses 16 separate conda environments to handle package conflicts:

- `bioinformatics tools`: SRA download, assembly, QC
- `ML/DL environments`: Separate environments for each model type
- `Analysis tools`: Feature selection, interpretability, batch correction

## 📁 Output Structure

```
results/
├── qc/                          # Quality control reports
├── features/                    # Processed feature matrices
│   ├── tree_models/            # Features for XGBoost/LightGBM
│   └── deep_models/            # K-mer and DNABERT datasets
├── models/                     # Trained model results
│   ├── xgboost/
│   ├── lightgbm/
│   ├── cnn/
│   ├── sequence_cnn/
│   └── dnabert/
└── interpretability/           # Cross-model analysis
    ├── consensus_features.csv
    ├── motif_analysis.json
    └── statistical_comparison.json
```

## 🧬 Data Sources

- **NCBI Pathogen Detection**: Public *K. pneumoniae* genomes
- **Temporal Split**: Pre-2023 training, 2023-24 testing
- **AST Phenotypes**: Extracted from metadata
- **Quality Filters**: Paired-end reads, contamination screening

## 📊 Shared Utilities

### Core Modules

- **`utils/cross_validation.py`**: Geographic-temporal CV strategy
- **`utils/class_balancing.py`**: Consistent class weighting
- **`utils/evaluation.py`**: Standardized metrics and success criteria
- **`utils/motif_analysis.py`**: Sequence motif extraction and analysis

### Benefits
- **Eliminates 1800+ lines** of duplicate code
- **Ensures consistency** across all 5 models
- **Standardizes evaluation** with automatic success checking
- **Maintainable codebase** with shared utilities

## 🚀 Deployment Options

### 📦 Container Deployment

The pipeline is fully containerized and supports multiple deployment scenarios:

| Platform | Use Case | Command | Documentation |
|----------|----------|---------|---------------|
| **Local Docker** | Development, testing | `make build && make run` | Built-in |
| **Google Cloud Run** | Serverless, auto-scaling | `./scripts/gcp/deploy-cloudrun.sh` | [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md) |
| **Google Kubernetes Engine** | Large-scale, GPU support | `./scripts/gcp/setup-gke.sh` | [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md) |
| **AWS/Azure** | Multi-cloud deployment | Standard Docker containers | [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md) |
| **HPC/Singularity** | Academic clusters | `singularity build amr.sif docker://...` | [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md) |

### 🎯 Quick Deployment Examples

```bash
# Local development
make build && make shell

# Google Cloud serverless
./scripts/gcp/deploy-cloudrun.sh --project my-project --allow-unauthenticated

# Production with CI/CD
gcloud builds submit --config scripts/gcp/cloudbuild.yaml \
  --substitutions=_DEPLOY_CLOUDRUN=true
```

### 💡 Deployment Features

- **Multi-stage builds** for optimized container size
- **Health checks** and monitoring integration
- **Persistent storage** support for large datasets
- **Auto-scaling** based on resource usage
- **CI/CD integration** with Cloud Build
- **Cost optimization** with preemptible instances

## 🔬 Research Questions

1. **Model Comparison**: How do the 5 models compare in F1 and balanced accuracy?
2. **Temporal Validation**: Do pre-2023 models retain ≥0.85 F1 on 2023-24 isolates?
3. **Interpretability**: Which genes, SNPs, or sequence motifs drive predictions?

## 📚 Citations

Based on genomic ML best practices from:
- **Statistical Testing**: DeLong et al. (1988), Demšar (2006)
- **Cross-Validation**: Genomic prediction literature
- **Deep Learning**: DNABERT-2 (Zhou et al.)
- **Class Imbalance**: Chen & Guestrin (XGBoost), Ke et al. (LightGBM)

## 🤝 Contributing

This project provides a reproducible template for AMR genomics studies. Contributions welcome:

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/improvement`)
3. **Commit** changes (`git commit -m 'Add improvement'`)
4. **Push** to branch (`git push origin feature/improvement`)
5. **Open** Pull Request

## 📄 License

Released under the [MIT License](LICENSE). Free for academic and commercial use.

## 🙏 Acknowledgments

- **NCBI Pathogen Detection** for public genomic data
- **HuggingFace** for DNABERT-2 model hosting
- **Snakemake** community for workflow management
- **Genomic ML** literature for methodological guidance

---

**🔬 For questions or collaboration**: [Contact Information]  
**📈 Latest Results**: Check `results/interpretability/` for current performance  
**🚀 Pipeline Status**: Run `snakemake --dry-run` to see next steps
# Interpretable Deep-Learning and Ensemble Models for Predicting Multidrug Resistance in *Klebsiella pneumoniae*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Snakemake](https://img.shields.io/badge/snakemake-≥7.32-brightgreen.svg)](https://snakemake.readthedocs.io)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)

A comprehensive, reproducible Snakemake workflow for genomic prediction of antimicrobial resistance (AMR) in *Klebsiella pneumoniae* using tree-based ensemble methods and deep learning architectures with temporal validation and interpretability analysis.

## 📋 Overview

This pipeline implements a 20-stage workflow comparing four machine learning architectures (XGBoost, LightGBM, 1D CNN, DNABERT-2) for predicting resistance to four critical antibiotic classes:
- **Carbapenems** (meropenem)
- **Cephalosporins** (ceftazidime)
- **Fluoroquinolones** (ciprofloxacin)
- **Aminoglycosides** (amikacin)

**Key Features:**
- ✅ Rigorous temporal validation (pre-2023 training → 2023-2024 testing)
- ✅ SHAP-based interpretability analysis
- ✅ Comprehensive quality control pipeline
- ✅ Automated feature engineering and selection
- ✅ Full reproducibility with conda environments
- ✅ Optimized for high-performance computing

## 🎓 Citation

If you use this workflow in your research, please cite:

```bibtex
@mastersthesis{nasirli2025kleb,
  author  = {Nasirli, Nasir},
  title   = {Interpretable Deep-Learning and Ensemble Models for Predicting 
             Multidrug Resistance in Klebsiella pneumoniae},
  school  = {University of Birmingham},
  year    = {2025},
  type    = {MSc Bioinformatics Dissertation}
}
```

See `CITATION.cff` for machine-readable citation metadata.

## 🚀 Quick Start

### Prerequisites

**Hardware Requirements:**
- **Recommended:** 32 vCPUs, 128GB RAM, 1TB SSD (Hetzner server or equivalent)
- **Minimum (Mac/laptop):** 8 cores, 16GB RAM (reduced parallelism, longer runtime)
- **Runtime:** 2-3 days for complete pipeline on recommended hardware
- **Storage:** ~700GB for metadata and results

**Software (choose one):**
- **Option A — Docker** (recommended, works on any OS including Mac ARM64):
  - [Docker Desktop](https://www.docker.com/products/docker-desktop/) ≥24.0
  - [Docker Compose](https://docs.docker.com/compose/) V2
- **Option B — Native Conda:**
  - [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/) (recommended)
  - [Snakemake](https://snakemake.readthedocs.io/) ≥7.32
  - Git

### Option A: Docker (Recommended)

Docker eliminates all dependency/platform issues — bioinformatics tools like SPAdes, freebayes, and kraken2 run in a Linux x86_64 container regardless of host OS.

```bash
# Clone repository
git clone https://github.com/NasirNesirli/kleb-amr-project.git
cd kleb-amr-project

# Build the image (first time takes ~30-60 min to create all conda envs)
docker compose build

# Run the full pipeline
docker compose up pipeline

# Or run individual stages
docker compose up preprocess     # Stages 1-5: QC + assembly
docker compose up train          # Stages 14-18: all models

# Interactive shell inside the container
docker compose run --rm dev

# Run tests
docker compose run --rm test
```

**Adjusting resources (edit `.env` or pass as environment):**
```bash
# Use 4 threads and limit to 32 GB RAM
THREADS=4 DOCKER_MEMORY=32G docker compose up pipeline

# On Mac with 8 cores:
THREADS=8 DOCKER_CPUS=8 DOCKER_MEMORY=16G docker compose up pipeline
```

**GPU training (NVIDIA only):**
```bash
# Requires nvidia-container-toolkit
docker compose --profile gpu up train-gpu
```

### Option B: Native Installation (Conda)

```bash
# Clone repository
git clone https://github.com/NasirNesirli/kleb-amr-project.git
cd kleb-amr-project

# Install Snakemake (if not already installed)
conda create -n snakemake -c conda-forge -c bioconda snakemake=7.32
conda activate snakemake
```

### Running the Pipeline

**Full Pipeline (20 stages):**
```bash
# Maximum parallelization strategy (recommended for 32 vCPU)
bash run_max_parallel.sh

# OR standard Snakemake execution
snakemake --use-conda --cores 32 --jobs 16
```

**Individual Stages:**
```bash
# Run specific stage independently
snakemake --use-conda --cores 32 -s rules/01_metadata.smk metadata_all
snakemake --use-conda --cores 32 -s rules/06_amr_analysis.smk amr_analysis_all
```

**Partial Workflows:**
```bash
# Preprocessing only (stages 1-5)
snakemake --use-conda --cores 32 preprocess

# Feature extraction (stages 6-10)
snakemake --use-conda --cores 32 feature_extraction

# Tree models only (stages 14-15)
snakemake --use-conda --cores 32 tree_models

# Deep learning models (stages 16-18)
snakemake --use-conda --cores 32 dl_models

# Interpretability analysis (stage 19)
snakemake --use-conda --cores 32 interpretability
```

### Running Tests

```bash
# Install test dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=utils --cov-report=term-missing
```


## 📊 Pipeline Stages

The workflow consists of 20 interconnected stages:

### Data Acquisition & QC (Stages 1-5)
1. **Metadata Processing** - Curate *K. pneumoniae* isolate metadata from NCBI
2. **Download** - Retrieve short-read sequencing data
3. **Pre-assembly QC** - FastQC, fastp trimming, quality filtering
4. **Assembly** - SPAdes genome assembly
5. **Post-assembly QC** - QUAST metrics, Kraken2 contamination screening

### Feature Engineering (Stages 6-10)
6. **AMR Analysis** - AMRFinderPlus resistance gene annotation
7. **SNP Analysis** - Snippy core-genome variant calling
8. **Feature Matrix** - Construct unified feature matrix
9. **Feature Selection** - Dimensionality reduction (1.2M → 325 features)
10. **Batch Correction** - Geographic-temporal batch effect removal

### Dataset Preparation (Stages 11-13)
11. **Training Data Preparation** - Temporal train/test splitting
12. **K-mer Datasets** - Tokenized sequences for CNN models
13. **DNABERT Datasets** - DNA transformer input preparation

### Model Training (Stages 14-18)
14. **XGBoost** - Gradient boosting with SHAP interpretability
15. **LightGBM** - Light gradient boosting machine
16. **1D CNN** - Convolutional neural network on k-mer spectra
17. **Sequence CNN** - CNN on raw DNA sequences
18. **DNABERT** - Fine-tuned DNA transformer model

### Analysis (Stages 19-20)
19. **Interpretability Analysis** - SHAP feature importance, biological validation
20. **Ensemble Analysis** - Model combination and comparative evaluation

## 📁 Project Structure

```
kleb-amr-project/
├── Snakefile                    # Master workflow orchestrator
├── run_max_parallel.sh          # Optimized parallel execution script
├── config/
│   └── config.yaml              # Pipeline configuration
├── rules/                       # Individual Snakemake rule modules (20 stages)
│   ├── 01_metadata.smk
│   ├── 02_download.smk
│   └── ...
├── scripts/                     # Python implementation scripts
│   ├── 01_metadata.py
│   ├── 14_train_xgboost.py
│   └── ...
├── envs/                        # Conda environment specifications
│   ├── assembly.yaml
│   ├── xgboost.yaml
│   └── ...
├── utils/                       # Shared utility modules
│   ├── cross_validation.py
│   ├── evaluation.py
│   └── ...
├── data/
│   └── metadata.csv             # User-provided NCBI metadata
├── results/                     # Auto-generated outputs
│   ├── qc/                      # Quality control reports
│   ├── features/                # Feature matrices
│   ├── models/                  # Trained model results
│   ├── interpretability/        # SHAP analysis outputs
│   └── ensemble/                # Ensemble evaluation
└── thesis/                      # Dissertation and documentation
```

## 🔧 Configuration

### Data Preparation

1. **Download metadata from NCBI Pathogen Detection:**
   - Visit: https://www.ncbi.nlm.nih.gov/pathogens/isolates/#taxgroup_name:%22Klebsiella%20pneumoniae%22
   - Filter for isolates with AMR susceptibility data
   - Download as CSV to `data/metadata.csv`

2. **Expected metadata format:**
   ```csv
   #Run;Collection date;AST phenotypes;Isolate;Location;Isolation source
   ```

### Pipeline Configuration

The `config/config.yaml` file contains default settings optimized for the workflow:

```yaml
antibiotics:
  - meropenem
  - ciprofloxacin
  - ceftazidime
  - amikacin

temporal_split_year: 2023
threads:
  assembly: 16
  mapping: 4
  annotation: 4
```

**Note:** Default configuration is optimized for the study design. Modification is typically not required.

## 📈 Key Results

Based on temporal validation (2023-2024 test set):

| Model | Meropenem | Ciprofloxacin | Ceftazidime | Amikacin |
|-------|-----------|---------------|-------------|----------|
| **XGBoost** | **0.824** | 0.787 | 0.800 | 0.500 |
| **LightGBM** | 0.583 | 0.827 | **0.857*** | 0.400 |
| 1D CNN | 0.091 | 0.825 | 0.778 | 0.000 |
| Sequence CNN | 0.095 | 0.369 | 0.536 | 0.013 |
| DNABERT-2 | 0.111 | 0.191 | 0.338 | 0.000 |

*Values shown are F1-scores. **Only model-antibiotic combination meeting F1≥0.85 clinical threshold.*

**Key Findings:**
- Tree-based models consistently outperform deep learning approaches
- SHAP analysis identifies biologically meaningful resistance determinants
- Temporal validation reveals robust generalization for tree models
- Dataset size limitations may impact deep learning performance

## 🧬 Interpretability Insights

SHAP analysis of XGBoost meropenem model revealed top predictive features:

1. `gene_parC_S80I` - Fluoroquinolone resistance mutation
2. `gene_oqxB19` - RND efflux pump component
3. `gene_aac(6')-Ib` - Aminoglycoside acetyltransferase
4. `gene_blaKPC-3` - KPC carbapenemase
5. `gene_ompK36_D135DGD` - Porin modification

These features align with established *K. pneumoniae* resistance mechanisms, validating model biological interpretability.

## 🛠️ Troubleshooting

### Independent Stage Execution

Each Snakemake rule file can be executed independently for debugging:

```bash
# Example: Run only AMR analysis
snakemake --use-conda --cores 8 -s rules/06_amr_analysis.smk amr_analysis_all

# Example: Run only XGBoost training
snakemake --use-conda --cores 32 -s rules/14_train_xgboost.smk train_xgboost_all
```

### Common Issues

**Memory errors during assembly:**
- Reduce parallel jobs: `--jobs 1` or `--jobs 2`
- SPAdes requires ~16GB per assembly

**Conda environment conflicts:**
- Use Mamba for faster dependency resolution: `snakemake --use-conda --conda-frontend mamba`

**Storage limitations:**
- Intermediate files can be large; ensure adequate disk space
- Consider cleaning up intermediate assemblies after QC

## 📚 Dependencies

All computational dependencies are managed through conda environments in `envs/`:

**Core Tools:**
- FastQC, fastp (QC)
- SPAdes (assembly)
- QUAST, Kraken2 (assembly QC)
- AMRFinderPlus (resistance annotation)
- Snippy, BWA (variant calling)
- scikit-learn, XGBoost, LightGBM (machine learning)
- PyTorch, Transformers (deep learning)
- SHAP (interpretability)

See individual YAML files for complete version specifications.


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Nasir Nasirli**  
MSc Bioinformatics, University of Birmingham  
Student ID: 2684202

## 🔗 Links

- **GitHub:** https://github.com/NasirNesirli/kleb-amr-project
- **NCBI Pathogen Detection:** https://www.ncbi.nlm.nih.gov/pathogens/
- **Dissertation:** See `thesis/final-dissertation.pdf`

## 📖 References

Complete references available in `thesis/final-dissertation.pdf`.

---

**Acknowledgments:** This work was conducted as part of the MSc Bioinformatics program at the University of Birmingham School of Biosciences (2025).
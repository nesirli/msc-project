"""
Master Snakefile for Klebsiella pneumoniae AMR Prediction Pipeline (19 Stages)
Run full pipeline: snakemake --use-conda --cores 8
Run specific step: snakemake --use-conda --cores 8 -s workflow/rules/01_metadata.smk
"""

configfile: "config/config.yaml"

# Rule order to resolve ambiguities
ruleorder: downsample_reads > fastp_trim

# Import all rule modules (18-stage reorganized pipeline)
include: "rules/01_metadata.smk"
include: "rules/02_download.smk"
include: "rules/03_preassembly_qc.smk"
include: "rules/04_assembly.smk"
include: "rules/05_postassembly_qc.smk"
include: "rules/06_amr_analysis.smk"
include: "rules/07_snp_analysis.smk"
include: "rules/08_feature_matrix.smk"
include: "rules/09_feature_selection.smk"
include: "rules/10_batch_correction.smk"
include: "rules/11_prepare_training_data.smk"
include: "rules/12_create_kmer_datasets.smk"
include: "rules/13_create_dnabert_datasets.smk"
include: "rules/14_train_xgboost.smk"
include: "rules/15_train_lightgbm.smk"
include: "rules/16_train_1dcnn.smk"
include: "rules/17_train_sequence_cnn.smk"
include: "rules/18_train_dnabert.smk"
include: "rules/19_interpretability_analysis.smk"
include: "rules/20_ensemble_analysis.smk"

# Define antibiotics
ANTIBIOTICS = config["antibiotics"]

rule all:
    input:
        # QC reports
        "results/qc/preassembly_multiqc.html",
        "results/qc/postassembly_multiqc.html",
        # Tree model datasets
        expand("results/features/tree_models/{antibiotic}_train_final.csv", antibiotic=ANTIBIOTICS),
        expand("results/features/tree_models/{antibiotic}_test_final.csv", antibiotic=ANTIBIOTICS),
        # Deep learning datasets
        expand("results/features/deep_models/{antibiotic}_kmer_train_final.npz", antibiotic=ANTIBIOTICS),
        expand("results/features/deep_models/{antibiotic}_dnabert_train_final.npz", antibiotic=ANTIBIOTICS),
        # Model results
        expand("results/models/xgboost/{antibiotic}_results.json", antibiotic=ANTIBIOTICS),
        expand("results/models/lightgbm/{antibiotic}_results.json", antibiotic=ANTIBIOTICS),
        expand("results/models/cnn/{antibiotic}_results.json", antibiotic=ANTIBIOTICS),
        expand("results/models/sequence_cnn/{antibiotic}_results.json", antibiotic=ANTIBIOTICS),
        expand("results/models/dnabert/{antibiotic}_results.json", antibiotic=ANTIBIOTICS),
        # Interpretability analysis
        expand("results/interpretability/{antibiotic}_interpretability.json", antibiotic=ANTIBIOTICS),
        # Ensemble analysis
        expand("results/ensemble/{antibiotic}_ensemble_analysis.json", antibiotic=ANTIBIOTICS),
        "results/ensemble/ensemble_summary_report.json"

# Preprocessing pipeline only (steps 1-5)
rule preprocess:
    input:
        "results/qc/preassembly_multiqc.html",
        "results/qc/postassembly_multiqc.html"

# Feature extraction pipeline (steps 6-10) 
rule feature_extraction:
    input:
        expand("results/features/{antibiotic}_train_batch_corrected.csv", antibiotic=ANTIBIOTICS),
        expand("results/features/{antibiotic}_test_batch_corrected.csv", antibiotic=ANTIBIOTICS)

# Balanced datasets for tree models (step 13)
rule balanced_datasets:
    input:
        expand("results/features/tree_models/{antibiotic}_train_final.csv", antibiotic=ANTIBIOTICS),
        expand("results/features/tree_models/{antibiotic}_test_final.csv", antibiotic=ANTIBIOTICS)

# Deep learning datasets (steps 14-15)
rule deep_datasets:
    input:
        expand("results/features/deep_models/{antibiotic}_kmer_train_final.npz", antibiotic=ANTIBIOTICS),
        expand("results/features/deep_models/{antibiotic}_dnabert_train_final.npz", antibiotic=ANTIBIOTICS)

# Tree models only (steps 14-15)
rule tree_models:
    input:
        expand("results/models/xgboost/{antibiotic}_results.json", antibiotic=ANTIBIOTICS),
        expand("results/models/lightgbm/{antibiotic}_results.json", antibiotic=ANTIBIOTICS)

# Deep learning models only (steps 16-18)
rule dl_models:
    input:
        expand("results/models/cnn/{antibiotic}_results.json", antibiotic=ANTIBIOTICS),
        expand("results/models/sequence_cnn/{antibiotic}_results.json", antibiotic=ANTIBIOTICS),
        expand("results/models/dnabert/{antibiotic}_results.json", antibiotic=ANTIBIOTICS)

# Interpretability analysis only (step 19)
rule interpretability:
    input:
        expand("results/interpretability/{antibiotic}_interpretability.json", antibiotic=ANTIBIOTICS)
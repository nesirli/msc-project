#!/bin/bash
# Maximum parallelization strategy optimized for 32 vCPU / 128GB RAM
# Efficiently distributes workload across independent stages
set -euo pipefail

echo "==================================================================="
echo "Maximum parallelization pipeline execution"
echo "Hardware: 32 vCPUs, 128GB RAM"
echo "==================================================================="

# Run all independent stages that don't depend on each other
echo "[Stage 1/8] Metadata & Download (parallel)..."
# Background jobs for independent stages
snakemake --use-conda --cores 32 --jobs 16 metadata_all &
PID1=$!
snakemake --use-conda --cores 32 --jobs 16 download_all &
PID2=$!

# Wait for prerequisites
wait $PID1 $PID2

# QC and Assembly (memory + CPU intensive)
echo "[Stage 2/8] Pre-assembly QC..."
snakemake --use-conda --cores 32 --jobs 16 preassembly_qc_all

echo "[Stage 3/8] Assembly (SPAdes uses 16 threads/job)..."
# SPAdes uses 16 threads per job, run 2 jobs in parallel with 128GB RAM
snakemake --use-conda --cores 32 --jobs 2 assembly_all

echo "[Stage 4/8] Post-assembly QC (Kraken2 memory-intensive)..."
# Kraken2 uses ~8-16GB RAM per job, with 128GB can run 4 parallel jobs
snakemake --use-conda --cores 32 --jobs 4 postassembly_qc_all

# Feature extraction - Run with increased parallelization
echo "[Stage 5/8] Feature extraction..."
echo "  -> AMR analysis (AMRFinder: 4 threads/job, 8 parallel jobs)..."
snakemake --use-conda --cores 32 --jobs 8 amr_analysis_all

echo "  -> SNP analysis (BWA: 4 threads/job, 8 parallel jobs)..."
snakemake --use-conda --cores 32 --jobs 8 snp_analysis_all

# Continue with feature engineering
echo "  -> Feature matrix, selection, batch correction..."
snakemake --use-conda --cores 32 --jobs 1 feature_matrix_all feature_selection_all batch_correction_all

# Dataset preparation - Run with full resources
echo "[Stage 6/8] Dataset preparation..."
echo "  -> Preparing training data..."
snakemake --use-conda --cores 32 --jobs 16 prepare_training_data_all

echo "  -> Creating k-mer datasets..."
snakemake --use-conda --cores 32 --jobs 1 create_kmer_datasets_all

echo "  -> Creating DNABERT datasets..."
snakemake --use-conda --cores 32 --jobs 1 create_dnabert_datasets_all

# Model training (each model uses 32 threads, run sequentially)
echo "[Stage 7/8] Model training (32 threads each, sequential)..."
echo "  -> XGBoost..."
snakemake --use-conda --cores 32 --jobs 1 train_xgboost_all

echo "  -> LightGBM..."
snakemake --use-conda --cores 32 --jobs 1 train_lightgbm_all

echo "  -> 1D CNN..."
snakemake --use-conda --cores 32 --jobs 1 train_1dcnn_all

echo "  -> Sequence CNN..."
snakemake --use-conda --cores 32 --jobs 1 train_sequence_cnn_all

echo "  -> DNABERT..."
snakemake --use-conda --cores 32 --jobs 1 train_dnabert_all

# Final analysis (lightweight, can run in parallel)
echo "[Stage 8/8] Final analysis (parallel)..."
snakemake --use-conda --cores 16 --jobs 8 interpretability_all &
PID8=$!
snakemake --use-conda --cores 16 --jobs 8 ensemble_analysis_all &
PID9=$!

wait $PID8 $PID9

echo "==================================================================="
echo "Maximum parallelization pipeline completed successfully!"
echo "==================================================================="
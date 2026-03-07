"""
Step 18: DNABERT Training with PyTorch
Trains transformer-based model on DNA sequences from balanced samples
Run independently: snakemake --use-conda --cores 8 -s rules/18_train_dnabert.smk
"""

configfile: "config/config.yaml"

rule train_dnabert:
    input:
        train="results/features/deep_models/{antibiotic}_dnabert_train_final.npz",
        test="results/features/deep_models/{antibiotic}_dnabert_test_final.npz",
        tokenizer="results/features/deep_models/{antibiotic}_dnabert_tokenizer.pkl"
    output:
        model="results/models/dnabert/{antibiotic}_model.pt",
        results="results/models/dnabert/{antibiotic}_results.json",
        attention="results/models/dnabert/{antibiotic}_attention.pkl",
        plots="results/models/dnabert/{antibiotic}_plots.png"
    params:
        cv_folds=config["models"]["cv_folds"],
        random_state=config["models"]["random_state"],
        epochs=20,
        batch_size=16,
        learning_rate=2e-5,
        d_model=128,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
        weight_decay=0.01,
        patience=5
    conda:
        "../envs/transformer.yaml"
    threads: config["resources"]["threads"]
    log:
        "logs/18_train_dnabert/{antibiotic}.log"
    script:
        "../scripts/18_train_dnabert.py"

rule train_dnabert_all:
    input:
        expand("results/models/dnabert/{antibiotic}_results.json", antibiotic=config["antibiotics"])
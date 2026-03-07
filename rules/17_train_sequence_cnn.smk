"""
Step 17: Train sequence-based 1D CNN using raw ACGT sequences
Run independently: snakemake --use-conda --cores 8 -s rules/17_train_sequence_cnn.smk
"""

configfile: "config/config.yaml"

rule train_sequence_cnn:
    input:
        train="results/features/tree_models/{antibiotic}_train_final.csv",
        test="results/features/tree_models/{antibiotic}_test_final.csv"
    output:
        model="results/models/sequence_cnn/{antibiotic}_model.pt",
        results="results/models/sequence_cnn/{antibiotic}_results.json",
        plots="results/models/sequence_cnn/{antibiotic}_plots.png"
    log:
        "logs/17_train_sequence_cnn/{antibiotic}.log"
    params:
        processed_dir="data/processed",
        n_reads_per_sample=1000,
        max_seq_length=200,
        epochs=50,
        batch_size=16,
        learning_rate=0.0001,
        dropout=0.5,
        weight_decay=0.001,
        patience=15,
        cv_folds=5,
        random_state=42
    conda:
        "../envs/cnn.yaml"
    threads: config["resources"]["threads"]
    script:
        "../scripts/17_train_sequence_cnn.py"

rule train_sequence_cnn_all:
    input:
        expand("results/models/sequence_cnn/{antibiotic}_results.json", 
               antibiotic=config["antibiotics"])
    output:
        touch("results/models/sequence_cnn/.all_complete")
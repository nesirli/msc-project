"""
Step 17: Train sequence-based 1D CNN using raw ACGT sequences
Run independently: snakemake --use-conda --cores 8 -s rules/17_train_sequence_cnn.smk train_sequence_cnn_all
"""

configfile: "config/config.yaml"

rule train_sequence_cnn_all:
    input:
        expand("results/models/sequence_cnn/{antibiotic}_results.json",
               antibiotic=config["antibiotics"])
    output:
        touch("results/models/sequence_cnn/.all_complete")

rule train_sequence_cnn:
    input:
        train="results/features/tree_models/{antibiotic}_train_final.csv",
        test="results/features/tree_models/{antibiotic}_test_final.csv"
    output:
        model="results/models/sequence_cnn/{antibiotic}_model.pt",
        results="results/models/sequence_cnn/{antibiotic}_results.json",
        plots="results/models/sequence_cnn/{antibiotic}_plots.png"
    params:
        processed_dir="data/processed",
        n_reads_per_sample=1000,
        max_seq_length=200,
        cv_folds=config["models"]["cv_folds"],
        random_state=config["models"]["random_state"],
        epochs=config["models"]["cnn"]["epochs"],
        batch_size=config["models"]["cnn"]["batch_size"],
        learning_rate=config["models"]["cnn"]["learning_rate"],
        dropout=config["models"]["cnn"]["dropout"],
        weight_decay=config["models"]["cnn"]["weight_decay"],
        patience=config["models"]["cnn"]["patience"]
    conda:
        "../envs/cnn.yaml"
    threads: config["resources"]["threads"]
    log:
        "logs/17_train_sequence_cnn/{antibiotic}.log"
    script:
        "../scripts/17_train_sequence_cnn.py"

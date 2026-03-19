"""
Step 16: 1D-CNN Training with PyTorch
Trains 1D convolutional neural network on k-mer frequency spectra
Run independently: snakemake --use-conda --cores 8 -s rules/16_train_1dcnn.smk train_1dcnn_all
"""

configfile: "config/config.yaml"

rule train_1dcnn_all:
    input:
        expand("results/models/cnn/{antibiotic}_results.json",
               antibiotic=config["antibiotics"])

rule train_1dcnn:
    input:
        train="results/features/deep_models/{antibiotic}_kmer_train_final.npz",
        test="results/features/deep_models/{antibiotic}_kmer_test_final.npz"
    output:
        model="results/models/cnn/{antibiotic}_model.pt",
        results="results/models/cnn/{antibiotic}_results.json",
        importance="results/models/cnn/{antibiotic}_importance.csv",
        plots="results/models/cnn/{antibiotic}_plots.png"
    params:
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
        "logs/16_train_1dcnn/{antibiotic}.log"
    script:
        "../scripts/16_train_1dcnn.py"

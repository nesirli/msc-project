"""
Step 15: LightGBM Training
5-fold nested CV with hyperparameter tuning using balanced data
Run independently: snakemake --use-conda --cores 8 -s rules/15_train_lightgbm.smk
"""

configfile: "config/config.yaml"

rule train_lightgbm:
    input:
        train="results/features/tree_models/{antibiotic}_train_final.csv",
        test="results/features/tree_models/{antibiotic}_test_final.csv"
    output:
        model="results/models/lightgbm/{antibiotic}_model.joblib",
        results="results/models/lightgbm/{antibiotic}_results.json",
        shap="results/models/lightgbm/{antibiotic}_shap.csv",
        plots="results/models/lightgbm/{antibiotic}_plots.png"
    params:
        cv_folds=config["models"]["cv_folds"],
        random_state=config["models"]["random_state"],
        param_grid=config["models"]["lightgbm"]
    conda:
        "../envs/lightgbm.yaml"
    threads: config["resources"]["threads"]
    log:
        "logs/15_train_lightgbm/{antibiotic}.log"
    script:
        "../scripts/15_train_lightgbm.py"

rule train_lightgbm_all:
    input:
        expand("results/models/lightgbm/{antibiotic}_results.json", antibiotic=config["antibiotics"])
"""
Step 14: XGBoost Training
5-fold nested CV with hyperparameter tuning using balanced data
Run independently: snakemake --use-conda --cores 8 -s rules/14_train_xgboost.smk
"""

configfile: "config/config.yaml"

rule train_xgboost:
    input:
        train="results/features/tree_models/{antibiotic}_train_final.csv",
        test="results/features/tree_models/{antibiotic}_test_final.csv"
    output:
        model="results/models/xgboost/{antibiotic}_model.joblib",
        results="results/models/xgboost/{antibiotic}_results.json",
        shap="results/models/xgboost/{antibiotic}_shap.csv",
        plots="results/models/xgboost/{antibiotic}_plots.png"
    params:
        cv_folds=config["models"]["cv_folds"],
        random_state=config["models"]["random_state"],
        param_grid=config["models"]["xgboost"]
    conda:
        "../envs/xgboost.yaml"
    threads: config["resources"]["threads"]
    log:
        "logs/14_train_xgboost/{antibiotic}.log"
    script:
        "../scripts/14_train_xgboost.py"

rule train_xgboost_all:
    input:
        expand("results/models/xgboost/{antibiotic}_results.json", antibiotic=config["antibiotics"])
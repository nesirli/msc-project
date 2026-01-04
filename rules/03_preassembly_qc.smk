"""
Step 03: Pre-assembly QC
FastQC -> MultiQC -> Fastp trimming -> MultiQC
Run independently: snakemake --use-conda --cores 8 -s rules/03_preassembly_qc.smk preassembly_qc_all
"""

import pandas as pd

configfile: "config/config.yaml"

rule fastqc_raw:
    input:
        r1="data/raw/{sample}_1.fastq.gz",
        r2="data/raw/{sample}_2.fastq.gz"
    output:
        html1="results/qc/fastqc_raw/{sample}_1_fastqc.html",
        html2="results/qc/fastqc_raw/{sample}_2_fastqc.html",
        zip1="results/qc/fastqc_raw/{sample}_1_fastqc.zip",
        zip2="results/qc/fastqc_raw/{sample}_2_fastqc.zip"
    params:
        outdir="results/qc/fastqc_raw"
    conda:
        "../envs/preassembly_qc.yaml"
    threads: 2
    log:
        "logs/03_preassembly_qc/fastqc_raw_{sample}.log"
    shell:
        """
        mkdir -p {params.outdir}
        fastqc -t {threads} -o {params.outdir} {input.r1} {input.r2} 2> {log}
        """

def get_all_raw_fastqc(wildcards):
    """Get all raw FastQC outputs"""
    try:
        train_df = pd.read_csv("results/features/metadata_train_processed.csv")
        test_df = pd.read_csv("results/features/metadata_test_processed.csv")
        
        # Use robust extraction like in 02_download.smk
        def extract_valid_runs(df):
            runs = []
            if 'Run' in df.columns:
                for run in df["Run"].astype(str):
                    run = str(run).strip()
                    if run.startswith('SRR') and 'SRR' in run:
                        import re
                        match = re.search(r'SRR\d+', run)
                        if match:
                            runs.append(match.group(0))
                    elif ',' in run and 'SRR' in run:
                        import re
                        match = re.search(r'SRR\d+', run)
                        if match:
                            runs.append(match.group(0))
            return runs
        
        train_samples = extract_valid_runs(train_df)
        test_samples = extract_valid_runs(test_df)
        all_samples = list(set(train_samples + test_samples))
        
        return expand("results/qc/fastqc_raw/{sample}_1_fastqc.zip", sample=all_samples)
    except Exception as e:
        print(f"Error in get_all_raw_fastqc: {e}")
        return []

rule multiqc_raw:
    input:
        get_all_raw_fastqc
    output:
        "results/qc/raw_multiqc.html"
    params:
        indir="results/qc/fastqc_raw",
        outdir="results/qc"
    conda:
        "../envs/preassembly_qc.yaml"
    log:
        "logs/03_preassembly_qc/multiqc_raw.log"
    shell:
        """
        multiqc {params.indir} -o {params.outdir} -n raw_multiqc 2> {log}
        rm -rf {params.indir} results/qc/raw_multiqc_data 2>> {log}
        """

rule fastp_trim:
    input:
        r1="data/raw/{sample}_1.fastq.gz",
        r2="data/raw/{sample}_2.fastq.gz",
        raw_qc="results/qc/raw_multiqc.html"
    output:
        r1="data/processed/{sample}_1.fastq.gz",
        r2="data/processed/{sample}_2.fastq.gz",
        html="results/qc/fastp/{sample}_fastp.html",
        json="results/qc/fastp/{sample}_fastp.json"
    params:
        min_len=config["qc"]["min_read_length"],
        min_qual=config["qc"]["min_quality"]
    conda:
        "../envs/preassembly_qc.yaml"
    threads: 4
    log:
        "logs/03_preassembly_qc/fastp_{sample}.log"
    shell:
        """
        mkdir -p results/qc/fastp
        fastp -i {input.r1} -I {input.r2} \
            -o {output.r1} -O {output.r2} \
            -h {output.html} -j {output.json} \
            -q {params.min_qual} -l {params.min_len} \
            -w {threads} 2> {log}
        rm -f {input.r1} {input.r2}
        """

rule fastqc_trimmed:
    input:
        r1="data/processed/{sample}_1.fastq.gz",
        r2="data/processed/{sample}_2.fastq.gz"
    output:
        html1="results/qc/fastqc_trimmed/{sample}_1_fastqc.html",
        html2="results/qc/fastqc_trimmed/{sample}_2_fastqc.html",
        zip1="results/qc/fastqc_trimmed/{sample}_1_fastqc.zip",
        zip2="results/qc/fastqc_trimmed/{sample}_2_fastqc.zip"
    params:
        outdir="results/qc/fastqc_trimmed"
    conda:
        "../envs/preassembly_qc.yaml"
    threads: 2
    log:
        "logs/03_preassembly_qc/fastqc_trimmed_{sample}.log"
    shell:
        """
        mkdir -p {params.outdir}
        fastqc -t {threads} -o {params.outdir} {input.r1} {input.r2} 2> {log}
        """

def get_all_fastqc_trimmed(wildcards):
    """Get all trimmed FastQC outputs"""
    try:
        train_df = pd.read_csv("results/features/metadata_train_processed.csv")
        test_df = pd.read_csv("results/features/metadata_test_processed.csv")
        
        # Use robust extraction like in 02_download.smk
        def extract_valid_runs(df):
            runs = []
            if 'Run' in df.columns:
                for run in df["Run"].astype(str):
                    run = str(run).strip()
                    if run.startswith('SRR') and 'SRR' in run:
                        import re
                        match = re.search(r'SRR\d+', run)
                        if match:
                            runs.append(match.group(0))
                    elif ',' in run and 'SRR' in run:
                        import re
                        match = re.search(r'SRR\d+', run)
                        if match:
                            runs.append(match.group(0))
            return runs
        
        train_samples = extract_valid_runs(train_df)
        test_samples = extract_valid_runs(test_df)
        all_samples = list(set(train_samples + test_samples))
        
        return expand("results/qc/fastqc_trimmed/{sample}_1_fastqc.zip", sample=all_samples)
    except Exception as e:
        print(f"Error in get_all_fastqc_trimmed: {e}")
        return []

rule multiqc_processed:
    input:
        get_all_fastqc_trimmed
    output:
        "results/qc/preassembly_multiqc.html"
    params:
        indir="results/qc/fastqc_trimmed",
        outdir="results/qc"
    conda:
        "../envs/preassembly_qc.yaml"
    log:
        "logs/03_preassembly_qc/multiqc_processed.log"
    shell:
        """
        multiqc {params.indir} -o {params.outdir} -n preassembly_multiqc 2> {log}
        rm -rf results/qc/fastqc_trimmed results/qc/fastp results/qc/preassembly_multiqc_data 2>> {log}
        """

rule preassembly_qc_all:
    input:
        raw_qc="results/qc/raw_multiqc.html",
        processed_qc="results/qc/preassembly_multiqc.html"
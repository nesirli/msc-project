"""
Step 07: SNP Calling
BWA alignment -> FreeBayes variant calling -> Quality filtering
Run independently: snakemake --use-conda --cores 8 -s rules/07_snp_analysis.smk
"""

import pandas as pd

configfile: "config/config.yaml"

rule download_reference:
    output:
        config["reference"]["fasta"]
    params:
        url=config["reference"]["url"]
    conda:
        "../envs/snp_analysis.yaml"
    log:
        "logs/07_snp_analysis/download_reference.log"
    shell:
        """
        wget -O {output}.gz {params.url} 2> {log}
        gunzip {output}.gz
        """

rule index_reference:
    input:
        config["reference"]["fasta"]
    output:
        config["reference"]["fasta"] + ".bwt",
        config["reference"]["fasta"] + ".fai"
    conda:
        "../envs/snp_analysis.yaml"
    log:
        "logs/07_snp_analysis/index_reference.log"
    shell:
        """
        bwa index {input} 2> {log}
        samtools faidx {input} 2>> {log}
        """

rule bwa_align:
    input:
        ref=config["reference"]["fasta"],
        ref_idx=config["reference"]["fasta"] + ".bwt",
        r1="data/processed/{sample}_1.fastq.gz",
        r2="data/processed/{sample}_2.fastq.gz"
    output:
        bam=temp("results/snp/bam/{sample}.sorted.bam"),
        bai=temp("results/snp/bam/{sample}.sorted.bam.bai")
    conda:
        "../envs/snp_analysis.yaml"
    threads: 4
    log:
        "logs/07_snp_analysis/bwa_{sample}.log"
    shell:
        """
        bwa mem -t {threads} {input.ref} {input.r1} {input.r2} 2> {log} | \
        samtools sort -@ {threads} -o {output.bam} - 2>> {log}
        samtools index {output.bam} 2>> {log}
        """

rule freebayes_call:
    input:
        ref=config["reference"]["fasta"],
        ref_fai=config["reference"]["fasta"] + ".fai",
        bam="results/snp/bam/{sample}.sorted.bam",
        bai="results/snp/bam/{sample}.sorted.bam.bai"
    output:
        vcf=temp("results/snp/vcf/{sample}.raw.vcf")
    conda:
        "../envs/snp_analysis.yaml"
    log:
        "logs/07_snp_analysis/freebayes_{sample}.log"
    shell:
        """
        freebayes -f {input.ref} {input.bam} > {output.vcf} 2> {log}
        """

rule filter_vcf:
    input:
        vcf="results/snp/vcf/{sample}.raw.vcf"
    output:
        vcf="results/snp/vcf/{sample}.filtered.vcf"
    params:
        min_qual=config["snp"]["min_qual"],
        min_depth=config["snp"]["min_depth"]
    conda:
        "../envs/snp_analysis.yaml"
    log:
        "logs/07_snp_analysis/filter_{sample}.log"
    shell:
        """
        bcftools filter -i 'QUAL>={params.min_qual} && INFO/DP>={params.min_depth}' \
            {input.vcf} -o {output.vcf} 2> {log}
        """

def check_pairing(sample):
    """Check if paired-end files are properly synchronized"""
    import gzip
    import os

    r1_path = f"data/processed/{sample}_1.fastq.gz"
    r2_path = f"data/processed/{sample}_2.fastq.gz"

    if not (os.path.exists(r1_path) and os.path.exists(r2_path)):
        return False

    try:
        # Read first read ID from each file
        with gzip.open(r1_path, 'rt') as f1, gzip.open(r2_path, 'rt') as f2:
            r1_line = f1.readline().strip()
            r2_line = f2.readline().strip()

            # Extract read number after the dot (e.g., SRR123.1 -> 1)
            if '.' in r1_line and '.' in r2_line:
                r1_num = r1_line.split()[0].split('.')[-1]
                r2_num = r2_line.split()[0].split('.')[-1]
                return r1_num == r2_num
            return True  # If format is different, assume OK
    except Exception:
        return False

def get_all_filtered_vcfs(wildcards):
    import os

    # If ready file exists, skip individual file checks
    if os.path.exists("results/snp/.snp_ready"):
        return []

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

        # Filter out samples with broken pairing
        valid_samples = [s for s in all_samples if check_pairing(s)]
        broken_count = len(all_samples) - len(valid_samples)
        if broken_count > 0:
            print(f"Note: Excluding {broken_count} samples with broken paired-end synchronization")

        return expand("results/snp/vcf/{sample}.filtered.vcf", sample=valid_samples)
    except Exception as e:
        print(f"Error in get_all_filtered_vcfs: {e}")
        return []

rule combine_snps:
    input:
        get_all_filtered_vcfs
    output:
        csv="results/snp/combined_snps.csv",
        ready=touch("results/snp/.snp_ready")
    conda:
        "../envs/snp_analysis.yaml"
    log:
        "logs/07_snp_analysis/combine_snps.log"
    script:
        "../scripts/07_snp_analysis.py"

rule snp_analysis_all:
    input:
        "results/snp/combined_snps.csv",
        "results/snp/.snp_ready"
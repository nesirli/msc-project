"""
Step 05: Post-assembly QC
QUAST for assembly metrics, Kraken2 for contamination screening
Run independently: snakemake --use-conda --cores 8 -s rules/05_postassembly_qc.smk
"""

import pandas as pd
import os

configfile: "config/config.yaml"

rule download_kraken2_db:
    output:
        directory(config["kraken2"]["db_path"])
    params:
        url=config["kraken2"]["db_url"]
    conda:
        "../envs/postassembly_qc.yaml"
    log:
        "logs/05_postassembly_qc/kraken2_db.log"
    shell:
        """
        mkdir -p {output}
        wget -O kraken2_db.tar.gz {params.url} 2> {log}
        tar -xzf kraken2_db.tar.gz -C {output} 2>> {log}
        rm kraken2_db.tar.gz
        """

rule quast_qc:
    input:
        assembly="data/assemblies/{sample}_assembled.fasta"
    output:
        report=temp("results/qc/quast/{sample}/report.tsv")
    params:
        outdir="results/qc/quast/{sample}",
        min_contig=config["qc"]["min_contig_length"]
    conda:
        "../envs/postassembly_qc.yaml"
    threads: 2
    log:
        "logs/05_postassembly_qc/quast_{sample}.log"
    shell:
        """
        quast.py {input.assembly} -o {params.outdir} \
            --min-contig {params.min_contig} -t {threads} 2> {log}
        """

rule kraken2_contamination:
    input:
        assembly="data/assemblies/{sample}_assembled.fasta",
        db=config["kraken2"]["db_path"]
    output:
        report=temp("results/qc/kraken2/{sample}_kraken2.report"),
        output=temp("results/qc/kraken2/{sample}_kraken2.out")
    conda:
        "../envs/postassembly_qc.yaml"
    threads: 4
    log:
        "logs/05_postassembly_qc/kraken2_{sample}.log"
    shell:
        """
        kraken2 --db {input.db} --threads {threads} \
            --report {output.report} --output {output.output} \
            {input.assembly} 2> {log}
        """

def get_all_quast_reports(wildcards):
    import glob
    assembly_files = glob.glob("data/assemblies/*_assembled.fasta")
    samples = [os.path.basename(f).replace("_assembled.fasta", "") for f in assembly_files]
    return expand("results/qc/quast/{sample}/report.tsv", sample=samples)

def get_all_kraken2_reports(wildcards):
    import glob
    assembly_files = glob.glob("data/assemblies/*_assembled.fasta")
    samples = [os.path.basename(f).replace("_assembled.fasta", "") for f in assembly_files]
    return expand("results/qc/kraken2/{sample}_kraken2.report", sample=samples)

rule multiqc_postassembly:
    input:
        quast=get_all_quast_reports,
        kraken=get_all_kraken2_reports
    output:
        "results/qc/postassembly_multiqc.html"
    params:
        indir="results/qc",
        outdir="results/qc"
    conda:
        "../envs/postassembly_qc.yaml"
    log:
        "logs/05_postassembly_qc/multiqc.log"
    shell:
        """
        multiqc {params.indir}/quast {params.indir}/kraken2 \
            -o {params.outdir} -n postassembly_multiqc 2> {log}
        rm -rf results/qc/quast results/qc/kraken2 results/qc/postassembly_multiqc_data 2>> {log}
        """

rule postassembly_qc_all:
    input:
        "results/qc/postassembly_multiqc.html"
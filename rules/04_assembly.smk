"""
Step 04: Genome Assembly with SPAdes
Downsample to 100x, assemble, rename and keep only final assembly
Run independently: snakemake --use-conda --cores 8 -s rules/04_assembly.smk assembly_all
"""

import pandas as pd

configfile: "config/config.yaml"

rule estimate_genome_size:
    output:
        "data/reference/genome_size.txt"
    shell:
        """
        echo "5500000" > {output}
        """

rule downsample_reads:
    input:
        r1="data/processed/{sample}_1.fastq.gz",
        r2="data/processed/{sample}_2.fastq.gz",
        genome_size="data/reference/genome_size.txt"
    output:
        r1=temp("data/processed/{sample}_ds_1.fastq.gz"),
        r2=temp("data/processed/{sample}_ds_2.fastq.gz")
    params:
        target_cov=config["qc"]["target_coverage"]
    conda:
        "../envs/assembly.yaml"
    log:
        "logs/04_assembly/downsample_{sample}.log"
    shell:
        """
        python scripts/04_assembly.py downsample \
            --r1 {input.r1} --r2 {input.r2} \
            --out_r1 {output.r1} --out_r2 {output.r2} \
            --genome_size $(cat {input.genome_size}) \
            --target_cov {params.target_cov} 2> {log}
        """

rule spades_assembly:
    input:
        r1="data/processed/{sample}_ds_1.fastq.gz",
        r2="data/processed/{sample}_ds_2.fastq.gz"
    output:
        assembly="data/assemblies/{sample}_assembled.fasta"
    params:
        outdir=temp(directory("data/assemblies/{sample}_spades"))
    conda:
        "../envs/assembly.yaml"
    threads: config["resources"]["threads"]
    log:
        "logs/04_assembly/spades_{sample}.log"
    shell:
        """
        # Check if we have sufficient reads for assembly
        reads_r1=$(zcat {input.r1} | wc -l)
        reads_r2=$(zcat {input.r2} | wc -l)
        min_reads=1000
        
        if [ $reads_r1 -lt $min_reads ] || [ $reads_r2 -lt $min_reads ]; then
            echo "ERROR: Insufficient reads for {wildcards.sample}: R1=$reads_r1 R2=$reads_r2 lines" | tee {log}
            echo "Minimum required: $min_reads lines (250 reads)" | tee -a {log}
            
            # Create empty assembly file to mark sample as failed
            echo ">failed_assembly_insufficient_data" > {output.assembly}
            echo "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN" >> {output.assembly}
            
            # Clean up
            rm -f {input.r1} {input.r2}
            exit 0
        fi
        
        spades.py -1 {input.r1} -2 {input.r2} \
            -o {params.outdir} \
            -t {threads} \
            --isolate \
            --only-assembler \
            -k 21,33,55 2> {log}
        
        if [ -f {params.outdir}/contigs.fasta ]; then
            cp {params.outdir}/contigs.fasta {output.assembly}
        else
            echo "ERROR: SPAdes failed to produce contigs.fasta for {wildcards.sample}" | tee -a {log}
            echo ">failed_assembly_spades_error" > {output.assembly}
            echo "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN" >> {output.assembly}
        fi
        
        rm -rf {params.outdir}
        
        # Immediately delete downsampled files to save disk space
        rm -f {input.r1} {input.r2}
        """

rule cleanup_raw_after_assembly:
    input:
        assembly="data/assemblies/{sample}_assembled.fasta"
    output:
        touch("data/assemblies/.{sample}_cleanup_done")
    shell:
        """
        rm -f data/raw/{wildcards.sample}_*.fastq.gz
        """

def get_all_assemblies(wildcards):
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
        
        return expand("data/assemblies/{sample}_assembled.fasta", sample=all_samples)
    except Exception as e:
        print(f"Error in get_all_assemblies: {e}")
        return []

rule assembly_all:
    input:
        get_all_assemblies
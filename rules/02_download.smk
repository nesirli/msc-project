"""
Step 02: Download SRA files
Downloads FASTQ files based on processed metadata
Run independently: snakemake --use-conda --cores 8 -s rules/02_download.smk
"""

import pandas as pd

configfile: "config/config.yaml"

def get_sample_ids(wildcards):
    """Get sample IDs from processed metadata"""
    try:
        train_df = pd.read_csv("results/features/metadata_train_processed.csv")
        test_df = pd.read_csv("results/features/metadata_test_processed.csv")
        
        # Robustly extract Run column - handle corrupted entries
        def extract_valid_runs(df):
            runs = []
            if 'Run' in df.columns:
                for run in df["Run"].astype(str):
                    # Clean and validate - extract SRR* pattern
                    run = str(run).strip()
                    if run.startswith('SRR') and 'SRR' in run:
                        # Extract just the SRR part if there's extra text
                        import re
                        match = re.search(r'SRR\d+', run)
                        if match:
                            runs.append(match.group(0))
                    elif ',' in run and 'SRR' in run:
                        # Handle corrupted CSV entries - extract SRR from malformed line
                        import re
                        match = re.search(r'SRR\d+', run)
                        if match:
                            runs.append(match.group(0))
            return runs
        
        train_samples = extract_valid_runs(train_df)
        test_samples = extract_valid_runs(test_df)
        
        all_samples = train_samples + test_samples
        unique_samples = list(set(all_samples))  # Remove duplicates
        
        print(f"Extracted {len(unique_samples)} valid sample IDs")
        return unique_samples
        
    except Exception as e:
        print(f"Error reading metadata files: {e}")
        return []

checkpoint download_metadata_ready:
    input:
        train="results/features/metadata_train_processed.csv",
        test="results/features/metadata_test_processed.csv"
    output:
        "results/.download_ready"
    shell:
        "touch {output}"

rule download_sample:
    input:
        ready="results/.download_ready"
    output:
        r1="data/raw/{sample}_1.fastq.gz",
        r2="data/raw/{sample}_2.fastq.gz"
    params:
        outdir="data/raw"
    conda:
        "../envs/download.yaml"
    threads: 2
    log:
        "logs/02_download/{sample}.log"
    shell:
        """
        # Create unique temp directory to prevent conflicts
        TEMP_DIR={params.outdir}/tmp_{wildcards.sample}_$$
        mkdir -p $TEMP_DIR
        
        # Download with retry logic
        for attempt in {{1..3}}; do
            if prefetch {wildcards.sample} -O $TEMP_DIR 2>> {log}; then
                break
            else
                echo "Attempt $attempt failed, retrying..." >> {log}
                rm -rf $TEMP_DIR/{wildcards.sample} 2>> {log}
                sleep $((attempt * 5))
            fi
        done
        
        # Extract reads with split-files for paired-end
        fasterq-dump $TEMP_DIR/{wildcards.sample}/{wildcards.sample}.sra \
            -O $TEMP_DIR -e {threads} --split-files 2>> {log}
        
        # For paired-end data, --split-files creates _1.fastq and _2.fastq
        # For single-end data, it creates .fastq (no suffix)
        if [ -f $TEMP_DIR/{wildcards.sample}.fastq ]; then
            # Single-end: rename to _1 and create empty _2
            mv $TEMP_DIR/{wildcards.sample}.fastq $TEMP_DIR/{wildcards.sample}_1.fastq
            touch $TEMP_DIR/{wildcards.sample}_2.fastq
        fi
        
        # Verify both files exist (should already exist for paired-end)
        if [ ! -f $TEMP_DIR/{wildcards.sample}_1.fastq ]; then
            echo "Error: Missing _1.fastq file for {wildcards.sample}" >> {log}
            exit 1
        fi
        if [ ! -f $TEMP_DIR/{wildcards.sample}_2.fastq ]; then
            echo "Error: Missing _2.fastq file for {wildcards.sample}" >> {log}
            echo "This should not happen for paired-end data" >> {log}
            exit 1
        fi
        
        # Compress and move to final location
        pigz -p {threads} $TEMP_DIR/{wildcards.sample}_1.fastq 2>> {log}
        pigz -p {threads} $TEMP_DIR/{wildcards.sample}_2.fastq 2>> {log}
        
        mv $TEMP_DIR/{wildcards.sample}_1.fastq.gz {output.r1}
        mv $TEMP_DIR/{wildcards.sample}_2.fastq.gz {output.r2}
        
        # Clean up
        rm -rf $TEMP_DIR 2>> {log}
        """

def aggregate_downloads(wildcards):
    """Aggregate all downloads after checkpoint"""
    checkpoint_output = checkpoints.download_metadata_ready.get(**wildcards).output[0]
    try:
        train_df = pd.read_csv("results/features/metadata_train_processed.csv")
        test_df = pd.read_csv("results/features/metadata_test_processed.csv")
        
        # Use same robust extraction function
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
        
        return expand("data/raw/{sample}_1.fastq.gz", sample=all_samples)
    except Exception as e:
        print(f"Error in aggregate_downloads: {e}")
        return []

rule download_all:
    input:
        aggregate_downloads
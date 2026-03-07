#!/usr/bin/env python3
"""
Assembly helper functions: downsampling reads to target coverage.
"""

import argparse
import subprocess
import gzip
from pathlib import Path

def count_bases(fastq_gz):
    """Count total bases in gzipped FASTQ file."""
    total_bases = 0
    with gzip.open(fastq_gz, 'rt') as f:
        for i, line in enumerate(f):
            if i % 4 == 1:  # Sequence line
                total_bases += len(line.strip())
    return total_bases

def downsample_reads(r1, r2, out_r1, out_r2, genome_size, target_cov):
    """Downsample paired reads to target coverage using seqtk."""
    # Count bases in R1
    total_bases = count_bases(r1) * 2  # Paired-end
    current_cov = total_bases / genome_size
    
    if current_cov <= target_cov:
        # No downsampling needed, just copy
        subprocess.run(f"cp {r1} {out_r1}", shell=True, check=True)
        subprocess.run(f"cp {r2} {out_r2}", shell=True, check=True)
        return
    
    # Calculate fraction to keep
    fraction = target_cov / current_cov
    seed = 42
    
    # Downsample with seqtk
    cmd1 = f"seqtk sample -s{seed} {r1} {fraction} | gzip > {out_r1}"
    cmd2 = f"seqtk sample -s{seed} {r2} {fraction} | gzip > {out_r2}"
    
    subprocess.run(cmd1, shell=True, check=True)
    subprocess.run(cmd2, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    # Downsample command
    ds_parser = subparsers.add_parser('downsample')
    ds_parser.add_argument('--r1', required=True)
    ds_parser.add_argument('--r2', required=True)
    ds_parser.add_argument('--out_r1', required=True)
    ds_parser.add_argument('--out_r2', required=True)
    ds_parser.add_argument('--genome_size', type=int, required=True)
    ds_parser.add_argument('--target_cov', type=int, default=100)
    
    args = parser.parse_args()
    
    if args.command == 'downsample':
        downsample_reads(args.r1, args.r2, args.out_r1, args.out_r2,
                        args.genome_size, args.target_cov)

if __name__ == "__main__":
    main()
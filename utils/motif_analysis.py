"""
Motif-level interpretability analysis for sequence models.
Extracts and analyzes important sequence patterns from CNN and DNABERT attention.
"""

import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
import re
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns


class SequenceMotifExtractor:
    """Extract important sequence motifs from CNN filters and attention patterns."""
    
    def __init__(self, min_motif_length=6, max_motif_length=15):
        self.min_motif_length = min_motif_length
        self.max_motif_length = max_motif_length
        self.nucleotides = ['A', 'T', 'G', 'C']
    
    def extract_cnn_motifs(self, model, sequences, importance_scores):
        """
        Extract motifs from CNN filters using gradient-weighted class activation.
        
        Args:
            model: Trained CNN model
            sequences: Input sequences (one-hot encoded)
            importance_scores: Feature importance scores
            
        Returns:
            Dictionary of motifs with their importance scores
        """
        motifs = defaultdict(list)
        
        # Get convolutional layers
        conv_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv1d)]
        
        if not conv_layers:
            print("No convolutional layers found in model")
            return {}
        
        # For each convolutional layer
        for layer_idx, conv_layer in enumerate(conv_layers):
            kernel_size = conv_layer.kernel_size[0]
            
            if kernel_size < self.min_motif_length:
                continue
            
            # Extract filters
            filters = conv_layer.weight.data.cpu().numpy()  # Shape: (out_channels, in_channels, kernel_size)
            
            # Convert filters to motif patterns
            for filter_idx in range(filters.shape[0]):
                filter_weights = filters[filter_idx]  # Shape: (in_channels, kernel_size)
                
                # For one-hot encoded sequences, in_channels = 4 (A, T, G, C)
                if filter_weights.shape[0] == 4:
                    motif_pattern = self._weights_to_motif(filter_weights)
                    motif_score = np.mean(np.abs(filter_weights))
                    
                    motifs[f'cnn_layer{layer_idx}_filter{filter_idx}'] = {
                        'pattern': motif_pattern,
                        'score': motif_score,
                        'length': kernel_size,
                        'type': 'CNN_filter'
                    }
        
        return dict(motifs)
    
    def _weights_to_motif(self, weights, threshold=0.3):
        """
        Convert CNN filter weights to nucleotide motif pattern.
        
        Args:
            weights: Filter weights (4, kernel_size) for ATGC
            threshold: Minimum weight to consider nucleotide active
            
        Returns:
            String representation of motif pattern
        """
        motif = []
        
        for pos in range(weights.shape[1]):
            pos_weights = weights[:, pos]
            max_idx = np.argmax(pos_weights)
            max_weight = pos_weights[max_idx]
            
            if max_weight > threshold:
                motif.append(self.nucleotides[max_idx])
            else:
                # Check for multiple active nucleotides
                active_nucs = [self.nucleotides[i] for i in range(4) if pos_weights[i] > threshold]
                if len(active_nucs) > 1:
                    motif.append(f"[{''.join(active_nucs)}]")
                else:
                    motif.append('N')  # Ambiguous position
        
        return ''.join(motif)
    
    def extract_kmer_motifs(self, important_kmers, k=11):
        """
        Extract meaningful motifs from important k-mers.
        
        Args:
            important_kmers: List of important k-mer features with scores
            k: K-mer length
            
        Returns:
            Dictionary of motif families and their characteristics
        """
        motif_families = defaultdict(list)
        
        for kmer_info in important_kmers:
            kmer = kmer_info['feature']
            score = kmer_info['importance']
            
            # Remove k-mer prefix if present
            if kmer.startswith('kmer_'):
                kmer_seq = kmer[5:]
            else:
                kmer_seq = kmer
            
            # Skip if not valid DNA sequence
            if not re.match(r'^[ATGC]+$', kmer_seq):
                continue
            
            # Find motif patterns within k-mer
            motifs = self._find_motifs_in_sequence(kmer_seq)
            
            for motif in motifs:
                motif_families[motif].append({
                    'kmer': kmer_seq,
                    'score': score,
                    'position': kmer_seq.find(motif)
                })
        
        # Summarize motif families
        motif_summary = {}
        for motif, occurrences in motif_families.items():
            if len(occurrences) >= 2:  # Motif appears in multiple k-mers
                total_score = sum(occ['score'] for occ in occurrences)
                motif_summary[motif] = {
                    'pattern': motif,
                    'frequency': len(occurrences),
                    'total_score': total_score,
                    'avg_score': total_score / len(occurrences),
                    'kmers': [occ['kmer'] for occ in occurrences],
                    'type': 'kmer_motif'
                }
        
        return motif_summary
    
    def _find_motifs_in_sequence(self, sequence):
        """Find potential motifs within a sequence."""
        motifs = []
        
        # Look for repeated patterns
        for length in range(self.min_motif_length, min(self.max_motif_length, len(sequence))):
            for start in range(len(sequence) - length + 1):
                motif = sequence[start:start + length]
                
                # Check if motif appears multiple times in the sequence or has biological relevance
                if self._is_meaningful_motif(motif):
                    motifs.append(motif)
        
        return list(set(motifs))  # Remove duplicates
    
    def _is_meaningful_motif(self, motif):
        """Check if a motif is potentially meaningful."""
        # Skip if too short or all same nucleotide
        if len(motif) < self.min_motif_length or len(set(motif)) == 1:
            return False
        
        # Check for balanced nucleotide composition (not too biased)
        composition = Counter(motif)
        entropy_score = entropy(list(composition.values()))
        
        # Entropy threshold to avoid very biased sequences
        if entropy_score < 0.5:
            return False
        
        # Check for known resistance-associated patterns (extend this list)
        resistance_patterns = [
            'TTGACA',  # -35 promoter consensus
            'TATAAT',  # -10 promoter consensus (Pribnow box)
            'TGTG',    # Common in β-lactamase promoters
            'CGCG',    # CpG-like patterns
            'GAATTC',  # EcoRI site (common in mobile elements)
            'AGATCT',  # BglII site
        ]
        
        # Give bonus for containing known patterns
        for pattern in resistance_patterns:
            if pattern in motif or motif in pattern:
                return True
        
        return True  # Accept other patterns that pass basic filters


class AttentionMotifExtractor:
    """Extract motifs from transformer attention patterns."""
    
    def __init__(self, min_attention_score=0.1):
        self.min_attention_score = min_attention_score
    
    def extract_attention_motifs(self, model, tokenizer, sequences, attention_scores):
        """
        Extract motifs from DNABERT attention patterns.
        
        Args:
            model: DNABERT model
            tokenizer: DNABERT tokenizer
            sequences: Input sequences
            attention_scores: Attention weights from model
            
        Returns:
            Dictionary of attention-based motifs
        """
        motifs = defaultdict(list)
        
        if not hasattr(model, 'attention_weights') and len(attention_scores) == 0:
            print("No attention scores available")
            return {}
        
        # Process attention patterns
        for seq_idx, sequence in enumerate(sequences):
            if seq_idx >= len(attention_scores):
                break
            
            attention = attention_scores[seq_idx]  # (n_heads, seq_len, seq_len)
            
            # Average across attention heads
            avg_attention = np.mean(attention, axis=0)
            
            # Find high-attention regions
            high_attention_regions = self._find_attention_regions(avg_attention)
            
            # Extract sequence motifs from high-attention regions
            tokens = tokenizer.tokenize(sequence)
            for region in high_attention_regions:
                start_pos, end_pos, attention_score = region
                
                # Extract motif from tokens
                motif_tokens = tokens[start_pos:end_pos]
                motif_sequence = ''.join(motif_tokens).replace('[UNK]', 'N')
                
                if len(motif_sequence) >= 6:  # Minimum motif length
                    motifs[motif_sequence].append({
                        'sequence_idx': seq_idx,
                        'attention_score': attention_score,
                        'position': start_pos,
                        'length': end_pos - start_pos
                    })
        
        # Summarize motifs
        motif_summary = {}
        for motif_seq, occurrences in motifs.items():
            if len(occurrences) >= 2:  # Appears in multiple sequences
                avg_attention = np.mean([occ['attention_score'] for occ in occurrences])
                motif_summary[motif_seq] = {
                    'pattern': motif_seq,
                    'frequency': len(occurrences),
                    'avg_attention': avg_attention,
                    'max_attention': max(occ['attention_score'] for occ in occurrences),
                    'type': 'attention_motif'
                }
        
        return motif_summary
    
    def _find_attention_regions(self, attention_matrix):
        """Find regions with high attention scores."""
        # Use diagonal attention (self-attention) as importance score
        self_attention = np.diag(attention_matrix)
        
        # Find peaks in attention
        high_attention_indices = np.where(self_attention > self.min_attention_score)[0]
        
        # Group consecutive high-attention positions into regions
        regions = []
        if len(high_attention_indices) > 0:
            start = high_attention_indices[0]
            end = start
            
            for i in range(1, len(high_attention_indices)):
                if high_attention_indices[i] == end + 1:
                    end = high_attention_indices[i]
                else:
                    # End of current region
                    avg_attention = np.mean(self_attention[start:end+1])
                    regions.append((start, end + 1, avg_attention))
                    start = high_attention_indices[i]
                    end = start
            
            # Add final region
            avg_attention = np.mean(self_attention[start:end+1])
            regions.append((start, end + 1, avg_attention))
        
        return regions


def analyze_cross_model_motifs(motif_results):
    """
    Analyze motifs across different model types to find consensus patterns.
    
    Args:
        motif_results: Dictionary with motifs from different models
        
    Returns:
        Cross-model motif analysis results
    """
    all_motifs = []
    
    # Collect all motifs
    for model_name, motifs in motif_results.items():
        for motif_id, motif_info in motifs.items():
            motif_info['model'] = model_name
            motif_info['motif_id'] = motif_id
            all_motifs.append(motif_info)
    
    # Find similar motifs across models
    motif_similarity = find_motif_similarity(all_motifs)
    
    # Rank motifs by cross-model importance
    consensus_motifs = rank_consensus_motifs(all_motifs, motif_similarity)
    
    return {
        'all_motifs': all_motifs,
        'motif_similarity': motif_similarity,
        'consensus_motifs': consensus_motifs
    }


def find_motif_similarity(motifs, similarity_threshold=0.8):
    """Find similar motifs across different models using sequence similarity."""
    similar_groups = []
    
    for i, motif1 in enumerate(motifs):
        for j, motif2 in enumerate(motifs[i+1:], i+1):
            similarity = calculate_sequence_similarity(motif1['pattern'], motif2['pattern'])
            
            if similarity >= similarity_threshold:
                similar_groups.append({
                    'motif1': motif1,
                    'motif2': motif2,
                    'similarity': similarity
                })
    
    return similar_groups


def calculate_sequence_similarity(seq1, seq2):
    """Calculate similarity between two DNA sequences."""
    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0
    
    # Simple alignment-free similarity
    longer_seq = seq1 if len(seq1) >= len(seq2) else seq2
    shorter_seq = seq2 if len(seq1) >= len(seq2) else seq1
    
    max_similarity = 0.0
    
    # Slide shorter sequence along longer sequence
    for offset in range(len(longer_seq) - len(shorter_seq) + 1):
        matches = 0
        for i in range(len(shorter_seq)):
            if longer_seq[offset + i] == shorter_seq[i]:
                matches += 1
        
        similarity = matches / len(shorter_seq)
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity


def rank_consensus_motifs(motifs, similarity_groups):
    """Rank motifs by their consensus importance across models."""
    motif_scores = defaultdict(list)
    
    # Collect scores for each unique motif pattern
    for motif in motifs:
        pattern = motif['pattern']
        score = motif.get('score', 0) or motif.get('avg_attention', 0) or motif.get('avg_score', 0)
        model = motif['model']
        
        motif_scores[pattern].append({
            'score': score,
            'model': model,
            'type': motif.get('type', 'unknown')
        })
    
    # Calculate consensus scores
    consensus_ranking = []
    for pattern, scores in motif_scores.items():
        total_score = sum(s['score'] for s in scores)
        num_models = len(set(s['model'] for s in scores))
        avg_score = total_score / len(scores)
        
        consensus_ranking.append({
            'pattern': pattern,
            'consensus_score': total_score * num_models,  # Weight by number of models
            'avg_score': avg_score,
            'num_models': num_models,
            'models': [s['model'] for s in scores]
        })
    
    # Sort by consensus score
    consensus_ranking.sort(key=lambda x: x['consensus_score'], reverse=True)
    
    return consensus_ranking


def visualize_motifs(motif_analysis, output_dir):
    """Create visualizations of motif analysis results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot consensus motifs
    consensus_motifs = motif_analysis['consensus_motifs'][:20]  # Top 20
    
    if consensus_motifs:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        patterns = [m['pattern'] for m in consensus_motifs]
        scores = [m['consensus_score'] for m in consensus_motifs]
        num_models = [m['num_models'] for m in consensus_motifs]
        
        bars = ax.barh(range(len(patterns)), scores, color=plt.cm.viridis([n/max(num_models) for n in num_models]))
        ax.set_yticks(range(len(patterns)))
        ax.set_yticklabels([p[:20] + '...' if len(p) > 20 else p for p in patterns])
        ax.set_xlabel('Consensus Importance Score')
        ax.set_title('Top Consensus Motifs Across Models')
        
        # Add colorbar for number of models
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=1, vmax=max(num_models)))
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Number of Models')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'consensus_motifs.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Motif visualizations saved to {output_dir}")
    
    return str(output_dir / 'consensus_motifs.png')
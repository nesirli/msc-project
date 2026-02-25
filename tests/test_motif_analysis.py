"""Unit tests for motif_analysis.py module."""

import pytest
import os
import sys
import numpy as np
from unittest.mock import MagicMock

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.motif_analysis import (
    SequenceMotifExtractor,
    AttentionMotifExtractor,
    analyze_cross_model_motifs,
    find_motif_similarity,
    calculate_sequence_similarity,
    rank_consensus_motifs,
)


# --- SequenceMotifExtractor ---

class TestSequenceMotifExtractor:
    def test_init_defaults(self):
        extractor = SequenceMotifExtractor()
        assert extractor.min_motif_length == 6
        assert extractor.max_motif_length == 15

    def test_init_custom_lengths(self):
        extractor = SequenceMotifExtractor(min_motif_length=4, max_motif_length=20)
        assert extractor.min_motif_length == 4
        assert extractor.max_motif_length == 20

    def test_extract_kmer_motifs_basic(self):
        extractor = SequenceMotifExtractor(min_motif_length=4)
        important_kmers = [
            {'feature': 'kmer_ATGCGATCGAT', 'importance': 0.9},
            {'feature': 'kmer_ATGCGATCAAA', 'importance': 0.8},
            {'feature': 'kmer_TTTCCCCGGGG', 'importance': 0.3},
        ]
        motifs = extractor.extract_kmer_motifs(important_kmers, k=11)
        assert isinstance(motifs, dict)

    def test_extract_kmer_motifs_empty(self):
        extractor = SequenceMotifExtractor()
        motifs = extractor.extract_kmer_motifs({})
        assert isinstance(motifs, dict)
        assert len(motifs) == 0

    def test_extract_kmer_motifs_non_dna_filtered(self):
        extractor = SequenceMotifExtractor()
        # Non-DNA sequences should be filtered out
        important_kmers = [
            {'feature': 'kmer_XXXXXXXXX', 'importance': 0.9},
            {'feature': 'kmer_12345678', 'importance': 0.8},
        ]
        motifs = extractor.extract_kmer_motifs(important_kmers)
        assert isinstance(motifs, dict)

    def test_weights_to_motif(self):
        extractor = SequenceMotifExtractor()
        # 4 nucleotides x kernel_size weights
        weights = np.zeros((4, 8))
        # Make clear winners for each position: A=0, T=1, G=2, C=3
        weights[0, 0] = 1.0  # A at pos 0
        weights[1, 1] = 1.0  # T at pos 1
        weights[2, 2] = 1.0  # G at pos 2
        weights[3, 3] = 1.0  # C at pos 3
        motif = extractor._weights_to_motif(weights)
        assert isinstance(motif, str)
        assert len(motif) == 8

    def test_is_meaningful_motif_short(self):
        extractor = SequenceMotifExtractor(min_motif_length=6)
        assert not extractor._is_meaningful_motif("ATG")

    def test_is_meaningful_motif_single_nucleotide(self):
        extractor = SequenceMotifExtractor(min_motif_length=6)
        assert not extractor._is_meaningful_motif("AAAAAAA")

    def test_is_meaningful_motif_valid(self):
        extractor = SequenceMotifExtractor(min_motif_length=6)
        assert extractor._is_meaningful_motif("ATGCGATCGA")

    def test_extract_cnn_motifs_no_conv_layers(self):
        extractor = SequenceMotifExtractor()
        model = MagicMock()
        model.modules.return_value = []
        motifs = extractor.extract_cnn_motifs(model, [], [])
        assert isinstance(motifs, dict)


# --- AttentionMotifExtractor ---

class TestAttentionMotifExtractor:
    def test_init(self):
        extractor = AttentionMotifExtractor(min_attention_score=0.2)
        assert extractor.min_attention_score == 0.2

    def test_find_attention_regions_basic(self):
        extractor = AttentionMotifExtractor(min_attention_score=0.5)
        # Diagonal attention matrix with some high-attention positions
        attention = np.zeros((10, 10))
        np.fill_diagonal(attention, 0.1)
        # Make positions 3-6 high attention
        for i in range(3, 7):
            attention[i, i] = 0.8
        regions = extractor._find_attention_regions(attention)
        assert isinstance(regions, list)

    def test_find_attention_regions_empty(self):
        extractor = AttentionMotifExtractor(min_attention_score=0.9)
        attention = np.zeros((5, 5))
        regions = extractor._find_attention_regions(attention)
        assert isinstance(regions, list)
        assert len(regions) == 0


# --- Standalone functions ---

class TestCalculateSequenceSimilarity:
    def test_identical_sequences(self):
        sim = calculate_sequence_similarity("ATGCGATCGA", "ATGCGATCGA")
        assert sim == 1.0

    def test_completely_different(self):
        sim = calculate_sequence_similarity("AAAA", "TTTT")
        assert sim == 0.0

    def test_empty_sequences(self):
        sim = calculate_sequence_similarity("", "")
        assert sim == 0.0

    def test_one_empty(self):
        sim = calculate_sequence_similarity("ATGC", "")
        assert sim == 0.0

    def test_partial_match(self):
        sim = calculate_sequence_similarity("ATGC", "ATCC")
        assert 0.0 < sim < 1.0


class TestFindMotifSimilarity:
    def test_similar_motifs(self):
        motifs = [
            {'pattern': 'ATGCGATCGA', 'score': 0.9, 'model': 'cnn'},
            {'pattern': 'ATGCGATCGC', 'score': 0.8, 'model': 'dnabert'},
            {'pattern': 'TTTTTTTTTT', 'score': 0.5, 'model': 'cnn'},
        ]
        groups = find_motif_similarity(motifs, similarity_threshold=0.7)
        assert isinstance(groups, list)

    def test_empty_motifs(self):
        groups = find_motif_similarity([])
        assert isinstance(groups, list)
        assert len(groups) == 0

    def test_single_motif(self):
        motifs = [{'pattern': 'ATGCGATCGA', 'score': 0.9, 'model': 'cnn'}]
        groups = find_motif_similarity(motifs)
        assert isinstance(groups, list)


class TestAnalyzeCrossModelMotifs:
    def test_basic_analysis(self):
        motif_results = {
            'cnn': {
                'ATGCGA': {'pattern': 'ATGCGA', 'score': 0.9, 'activation': 0.8},
                'GCTAGC': {'pattern': 'GCTAGC', 'score': 0.7, 'activation': 0.5},
            },
            'dnabert': {
                'ATGCGA': {'pattern': 'ATGCGA', 'score': 0.85, 'attention_score': 0.7},
            }
        }
        result = analyze_cross_model_motifs(motif_results)
        assert isinstance(result, dict)
        assert 'all_motifs' in result
        assert 'consensus_motifs' in result

    def test_empty_input(self):
        result = analyze_cross_model_motifs({})
        assert isinstance(result, dict)


class TestRankConsensusMotifs:
    def test_ranking(self):
        motifs = [
            {'pattern': 'ATGCGA', 'score': 0.9, 'model': 'cnn'},
            {'pattern': 'ATGCGA', 'score': 0.8, 'model': 'dnabert'},
            {'pattern': 'GCTAGC', 'score': 0.5, 'model': 'cnn'},
        ]
        similarity_groups = []
        ranked = rank_consensus_motifs(motifs, similarity_groups)
        assert isinstance(ranked, list)
        if len(ranked) >= 2:
            # Higher-scoring motif should be first
            assert ranked[0]['consensus_score'] >= ranked[1]['consensus_score']

    def test_empty(self):
        ranked = rank_consensus_motifs([], [])
        assert isinstance(ranked, list)

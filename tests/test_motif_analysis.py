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
    @pytest.mark.parametrize(
        "min_len, max_len",
        [
            pytest.param(6, 15, id="defaults"),
            pytest.param(4, 20, id="custom"),
        ],
    )
    def test_init_lengths(self, min_len, max_len):
        extractor = SequenceMotifExtractor(min_motif_length=min_len, max_motif_length=max_len)
        assert extractor.min_motif_length == min_len
        assert extractor.max_motif_length == max_len

    def test_extract_kmer_motifs_basic(self):
        extractor = SequenceMotifExtractor(min_motif_length=4)
        important_kmers = [
            {'feature': 'kmer_ATGCGATCGAT', 'importance': 0.9},
            {'feature': 'kmer_ATGCGATCAAA', 'importance': 0.8},
            {'feature': 'kmer_TTTCCCCGGGG', 'importance': 0.3},
        ]
        motifs = extractor.extract_kmer_motifs(important_kmers, k=11)
        assert isinstance(motifs, dict)

    @pytest.mark.parametrize(
        "kmers",
        [
            pytest.param({}, id="empty_dict"),
            pytest.param(
                [
                    {'feature': 'kmer_XXXXXXXXX', 'importance': 0.9},
                    {'feature': 'kmer_12345678', 'importance': 0.8},
                ],
                id="non_dna",
            ),
        ],
    )
    def test_extract_kmer_motifs_edge_cases(self, kmers):
        extractor = SequenceMotifExtractor()
        motifs = extractor.extract_kmer_motifs(kmers)
        assert isinstance(motifs, dict)

    def test_weights_to_motif(self):
        extractor = SequenceMotifExtractor()
        weights = np.zeros((4, 8))
        weights[0, 0] = 1.0  # A at pos 0
        weights[1, 1] = 1.0  # T at pos 1
        weights[2, 2] = 1.0  # G at pos 2
        weights[3, 3] = 1.0  # C at pos 3
        motif = extractor._weights_to_motif(weights)
        assert isinstance(motif, str)
        assert len(motif) == 8

    @pytest.mark.parametrize(
        "motif, expected",
        [
            pytest.param("ATG", False, id="too_short"),
            pytest.param("AAAAAAA", False, id="single_nucleotide_repeat"),
            pytest.param("ATGCGATCGA", True, id="valid"),
        ],
    )
    def test_is_meaningful_motif(self, motif, expected):
        extractor = SequenceMotifExtractor(min_motif_length=6)
        assert extractor._is_meaningful_motif(motif) == expected

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

    @pytest.mark.parametrize(
        "threshold, fill_value, high_range, expected_empty",
        [
            pytest.param(0.5, 0.1, (3, 7), False, id="has_regions"),
            pytest.param(0.9, 0.0, None, True, id="no_regions"),
        ],
    )
    def test_find_attention_regions(self, threshold, fill_value, high_range, expected_empty):
        extractor = AttentionMotifExtractor(min_attention_score=threshold)
        size = 10 if high_range else 5
        attention = np.zeros((size, size))
        np.fill_diagonal(attention, fill_value)
        if high_range:
            for i in range(high_range[0], high_range[1]):
                attention[i, i] = 0.8
        regions = extractor._find_attention_regions(attention)
        assert isinstance(regions, list)
        if expected_empty:
            assert len(regions) == 0


# --- Standalone functions ---

class TestCalculateSequenceSimilarity:
    @pytest.mark.parametrize(
        "seq1, seq2, expected",
        [
            pytest.param("ATGCGATCGA", "ATGCGATCGA", 1.0, id="identical"),
            pytest.param("AAAA", "TTTT", 0.0, id="completely_different"),
            pytest.param("", "", 0.0, id="both_empty"),
            pytest.param("ATGC", "", 0.0, id="one_empty"),
        ],
    )
    def test_known_similarity(self, seq1, seq2, expected):
        assert calculate_sequence_similarity(seq1, seq2) == expected

    def test_partial_match(self):
        sim = calculate_sequence_similarity("ATGC", "ATCC")
        assert 0.0 < sim < 1.0


class TestFindMotifSimilarity:
    @pytest.mark.parametrize(
        "motifs, expected_empty",
        [
            pytest.param([], True, id="empty"),
            pytest.param(
                [{'pattern': 'ATGCGATCGA', 'score': 0.9, 'model': 'cnn'}],
                False,
                id="single",
            ),
            pytest.param(
                [
                    {'pattern': 'ATGCGATCGA', 'score': 0.9, 'model': 'cnn'},
                    {'pattern': 'ATGCGATCGC', 'score': 0.8, 'model': 'dnabert'},
                    {'pattern': 'TTTTTTTTTT', 'score': 0.5, 'model': 'cnn'},
                ],
                False,
                id="similar_group",
            ),
        ],
    )
    def test_find_motif_similarity(self, motifs, expected_empty):
        groups = find_motif_similarity(motifs, similarity_threshold=0.7) if motifs else find_motif_similarity(motifs)
        assert isinstance(groups, list)
        if expected_empty:
            assert len(groups) == 0


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
        ranked = rank_consensus_motifs(motifs, [])
        assert isinstance(ranked, list)
        if len(ranked) >= 2:
            assert ranked[0]['consensus_score'] >= ranked[1]['consensus_score']

    def test_empty(self):
        ranked = rank_consensus_motifs([], [])
        assert isinstance(ranked, list)

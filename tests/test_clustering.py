"""Mixture selection should find evidence, including evidence for no split."""

import numpy as np
import pytest

from kicks.analysis.latents import analyze_clusters, select_n_clusters


def test_single_gaussian_is_not_forced_into_multiple_clusters():
    points = np.random.default_rng(8).normal(size=(400, 3))
    result = analyze_clusters(points, max_k=4)
    assert result['diagnostics']['selected_k'] == 1
    assert result['diagnostics']['silhouette'] is None
    assert np.all(result['entropy'] == 0)
    assert select_n_clusters(points, max_k=3)[0] == 1


def test_separated_groups_and_constant_dimensions_are_reproducible():
    rng = np.random.default_rng(17)
    points = np.concatenate([rng.normal(-4, .4, (180, 3)), rng.normal(4, .4, (120, 3))])
    points = np.column_stack([points, np.ones(len(points))])
    first = analyze_clusters(points, max_k=4)
    second = analyze_clusters(points, max_k=4)
    assert first['diagnostics']['selected_k'] == 2
    assert first['diagnostics']['silhouette'] > .8
    assert first['diagnostics']['at_search_boundary'] is False
    np.testing.assert_array_equal(first['labels'], second['labels'])
    np.testing.assert_allclose(first['probabilities'].sum(axis=1), 1)
    assert (first['labels'][:180] == 0).all()
    assert first['projection'].shape == (300, 3)
    assert np.isfinite(first['projection']).all()
    assert len(first['diagnostics']['candidates']) == 8


def test_small_corpus_and_search_boundary_are_explicit():
    result = analyze_clusters(np.array([[0.], [1.], [2.]]), max_k=1)
    assert result['projection'].shape == (3, 3)
    assert result['diagnostics']['at_search_boundary'] is True
    assert result['diagnostics']['selected_k'] == 1


def test_pca_does_not_truncate_informative_dimensions_at_an_arbitrary_cap():
    points = np.random.default_rng(21).normal(size=(100, 20))
    result = analyze_clusters(points, max_k=1)
    assert result['diagnostics']['retained_variance'] >= .95
    assert result['diagnostics']['dimensions'] > 12


@pytest.mark.parametrize('points', [np.ones((10, 3)), np.zeros((1, 3)), np.array([[0.], [np.nan], [1.]])])
def test_invalid_inputs_fail_clearly(points):
    with pytest.raises(ValueError):
        analyze_clusters(points)

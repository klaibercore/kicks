"""The published analysis must describe the corpus without identifying its files."""

import json

from kicks.analysis.publish import publish_analysis, publish_report
from kicks.instruments import get_profile


def _report(profile):
    keys = profile.descriptor_keys
    return {
        "instrument": profile.name,
        "descriptor_keys": keys,
        "descriptor_labels": profile.descriptor_labels,
        "pca_variance_explained": [0.5, 0.3, 0.1],
        "pca_source": "descriptors_zscore",
        "n_clusters": 2,
        "corpus": {"sample_rate": 44100, "audio_length_ms": 1486.07, "n_total": 2, "data_dir": "data/secret"},
        "samples": [
            {
                "sample_idx": i,
                "filename": f"Vendor Pack {i}.wav",
                "original_path": f"data/secret/Vendor Pack {i}.wav",
                "cluster": i,
                "descriptors": {k: 0.123456789 for k in keys},
                "probs": [0.9, 0.1] if i == 0 else [0.2, 0.8],
                "duration_ms": 123.456789,
                "pc1": 0.1, "pc2": 0.2, "pc3": 0.3,
            }
            for i in range(2)
        ],
        "cluster_averages": {"0": [0.0] * 32, "1": [0.0] * 32},
        "pc_names": [{"name": "Sub", "descriptor": keys[0], "correlation": 0.9}],
        "pca_loadings": {"pc1": {k: 0.1 for k in keys}, "pc2": {k: 0.1 for k in keys}, "pc3": {k: 0.1 for k in keys}},
        "pc_descriptor_correlations": {"pc1": {k: 0.1 for k in keys}},
        "descriptor_correlations": {k: {j: 1.0 for j in keys} for k in keys},
        "cluster_profiles": {"0": {"count": 1, **{k: 0.1 for k in keys}}, "1": {"count": 1, **{k: 0.2 for k in keys}}},
        "descriptor_stats": {k: {"mean": 0.1, "std": 0.1, "min": 0.0, "max": 1.0} for k in keys},
    }


def test_published_report_has_no_file_identity():
    profile = get_profile("kick")
    published = publish_report(_report(profile))
    text = json.dumps(published)
    assert "Vendor Pack" not in text
    assert "data/secret" not in text
    assert "filename" not in text and "original_path" not in text
    assert "cluster_averages" not in published
    sample = published["samples"][0]
    assert set(sample) == {"sample_idx", "cluster", "duration_ms", "descriptors", "pc1", "pc2", "pc3", "confidence"}
    assert sample["confidence"] == 0.9
    assert sample["descriptors"][profile.descriptor_keys[0]] == 0.1235
    assert published["descriptor_docs"][profile.descriptor_keys[0]]


def test_publish_writes_index_and_skips_missing(tmp_path, monkeypatch):
    profile = get_profile("snare")
    out_root = tmp_path / "output"
    (out_root / "snare").mkdir(parents=True)
    (out_root / "snare" / "cluster_analysis.json").write_text(json.dumps(_report(profile)))
    monkeypatch.setenv("KICKS_OUTPUT_DIR", str(out_root))

    web = tmp_path / "web"
    index = publish_analysis(out_dir=str(web))

    names = [entry["name"] for entry in index]
    assert names == ["snare"]  # kick and hihat have no report here and are skipped
    assert (web / "snare.json").exists()
    assert json.loads((web / "index.json").read_text())["instruments"][0]["n_samples"] == 2


def test_clustering_evidence_and_projected_coordinates_are_published():
    report = _report(get_profile("kick"))
    report["schema_version"] = 2
    report["generated_at"] = "2026-09-16T12:00:00+00:00"
    report["clustering"] = {"selected_k": 2, "silhouette": 0.1234567,
                            "candidates": [{"k": 2, "covariance": "full", "bic": 123.456789, "converged": True}]}
    report["latent_projection"] = {"method": "pca", "variance_explained": [.6, .2, .1]}
    report["cluster_details"] = {"0": {"representative_idx": 0, "mean_confidence": .9, "ambiguous_count": 0}}
    report["samples"][0].update(latent1=1.12345678, latent2=-2.0, latent3=0.0, entropy=.12345678)
    published = publish_report(report)
    assert published["clustering"]["silhouette"] == .1235
    assert published["clustering"]["candidates"][0]["bic"] == 123.4568
    assert published["latent_projection"] == report["latent_projection"]
    assert published["samples"][0]["latent1"] == 1.1235
    assert published["samples"][0]["entropy"] == .1235
    assert published["cluster_details"]["0"]["representative_idx"] == 0
    assert "Vendor Pack" not in json.dumps(published)

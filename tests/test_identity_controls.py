"""Identity retention and reproducibility, independent of trained checkpoints."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from starlette.requests import Request

from kicks.analysis.basis import SliderBasis, slider_positions_to_axis_values
from kicks.synthesis.identity import IDENTITY_RADIUS, IdentityBasis


@pytest.fixture
def fitted():
    z = np.random.default_rng(71).normal(size=(400, 8))

    def response(x):
        return np.column_stack([x[:, 0] + .2 * x[:, 1], x[:, 2] - .3 * x[:, 3]])

    basis = IdentityBasis(z, response(z))
    return basis, response


def test_controls_retain_the_same_textures_null_space(fitted):
    basis, response = fitted
    for seed in (0, 7, 4294967295):
        anchor, directions = basis.local_directions(response, seed)
        neutral = np.eye(directions.shape[0]) - directions @ directions.T
        for target in ([.2, -.2], [-.3, .3], [1e6, -1e6]):
            z = basis.solve(target, response, seed=seed)
            step = ((z - anchor) @ basis.transform_inverse.T)[0]
            assert np.linalg.norm(step) <= IDENTITY_RADIUS + 1e-8
            np.testing.assert_allclose(neutral @ step, 0, atol=1e-8)
            np.testing.assert_array_equal(z, basis.solve(target, response, seed=seed))
        # Nearby descriptor targets must still respond to control changes.
        initial = response(anchor)[0]
        changed = response(basis.solve(initial + [.1, -.1], response, seed=seed))[0]
        np.testing.assert_allclose(changed, initial + [.1, -.1], atol=.005)


def test_seed_selects_an_encoded_texture(fitted):
    basis, _ = fitted
    anchors = [basis.anchor_index(seed) for seed in range(20)]
    assert len(set(anchors)) > 10
    for seed, idx in enumerate(anchors):
        np.testing.assert_array_equal(basis.solve([0, 0], seed=seed), basis.support_z[idx][None])


def test_percentile_controls_center_on_median_and_move_monotonically(fitted):
    core, _ = fitted
    # Highly skewed duration: midpoint of endpoints would be 500, not 19.
    points = np.array([[0, 1], [4, 10], [10, 19], [20, 41], [40, 999]])
    basis = SliderBasis(core, core.support_d, ["tone", "decay"], [0, 1], [40, 999], control_points=points)
    assert slider_positions_to_axis_values([.5, .5], basis) == [10, 19]
    curve = np.array([slider_positions_to_axis_values([p, p], basis) for p in np.linspace(0, 1, 101)])
    assert (np.diff(curve, axis=0) >= 0).all()
    np.testing.assert_array_equal(curve[[0, 25, 50, 75, 100]], points)
    with pytest.raises(ValueError):
        slider_positions_to_axis_values([np.nan, .5], basis)


@pytest.mark.parametrize("seed", ["-1", "4294967296", "1.0", "1e3", "nan", "inf", "", "abc"])
def test_api_rejects_invalid_texture_seeds(seed):
    from fastapi import HTTPException
    from kicks.api.app import _texture_seed
    with pytest.raises(HTTPException) as error:
        _texture_seed(Request({"type": "http", "query_string": f"seed={seed}".encode()}))
    assert error.value.status_code == 422


def test_render_cache_separates_variations_and_reuses_identical_audio(monkeypatch):
    import importlib
    api = importlib.import_module("kicks.api.app")
    from kicks.api.middleware import LRUCache
    from kicks.instruments import get_profile

    inst = SimpleNamespace(model=object(), basis=SimpleNamespace(names=["Body"]),
                           profile=get_profile("snare"), response=None)
    calls = []
    vocoder = object()
    monkeypatch.setattr(api.state, "vocoder_for", lambda _: vocoder)
    monkeypatch.setattr(api, "render_cache", LRUCache(max_size=32))

    def render(*args, seed):
        calls.append(seed)
        return torch.zeros(1, 1, 2, 2), torch.full((1, 32), seed / 10)

    monkeypatch.setattr(api, "render_controls", render)
    def request(seed):
        return Request({"type": "http", "query_string": f"body=0.5&seed={seed}".encode()})
    a = api._synthesize(request(1), inst)
    b = api._synthesize(request(2), inst)
    again = api._synthesize(request(1), inst)
    assert calls == [1, 2]
    assert not torch.equal(a[1], b[1])
    torch.testing.assert_close(a[1], again[1], rtol=0, atol=0)


def test_identity_cache_roundtrip_and_checkpoint_invalidation(tmp_path, monkeypatch, fitted):
    import kicks.synthesis.identity as identity
    from kicks.instruments import get_profile
    core, _ = fitted
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"model-a")
    dataset = SimpleNamespace(paths=[])
    calls = []
    monkeypatch.setattr(identity, "extract_latents", lambda *a: (calls.append(True) or core.support_z, None))
    monkeypatch.setattr(identity, "_decoded_descriptors", lambda *a: (core.support_z, core.support_d))
    first = identity.fit_identity_basis(None, dataset, get_profile("snare"), "cpu", checkpoint=checkpoint)
    second = identity.fit_identity_basis(None, dataset, get_profile("snare"), "cpu", checkpoint=checkpoint)
    assert len(calls) == 1
    np.testing.assert_array_equal(first.basis.support_z, second.basis.support_z)
    np.testing.assert_array_equal(first.control_points, second.control_points)
    checkpoint.write_bytes(b"model-b")
    identity.fit_identity_basis(None, dataset, get_profile("snare"), "cpu", checkpoint=checkpoint)
    assert len(calls) == 2


def test_unreachable_targets_do_not_overwrite_waveform_structure():
    from kicks.analysis.descriptors import descriptor_tensor
    from kicks.audio.constants import SAMPLE_RATE
    from kicks.audio.controls import correct_waveform
    from kicks.audio.mel import spectrogram
    from kicks.instruments import get_profile

    profile = get_profile("snare")
    t = torch.arange(65536) / SAMPLE_RATE
    noise = torch.randn(len(t), generator=torch.Generator().manual_seed(16))
    hit = (torch.sin(2 * torch.pi * 240 * t) + .4 * noise) * torch.exp(-t / .04)
    measured = descriptor_tensor(spectrogram(hit[None]), profile)[0].numpy()
    target = measured + [60, -60, 60, -60, 400]
    audio = correct_waveform(hit, target, np.array([40, 40, 40, 40, 200]), profile,
                              max_gain_db=6, regularization=.01)
    assert torch.isfinite(audio).all() and audio.abs().max() <= 1
    # Impossible descriptor combinations must leave the original oscillation
    # and noise pattern recognizable; fitting numbers cannot replace the hit.
    similarity = torch.dot(hit, audio) / (hit.norm() * audio.norm())
    assert similarity > .7
    assert audio[-4000:].square().mean() < 1e-7

"""The shared studio, export and control-audit rendering path."""

from __future__ import annotations

import numpy as np
import torch

from ..analysis.basis import slider_positions_to_axis_values
from ..audio.controls import correct_waveform
from ..audio.vocoder import spec_to_audio
from .identity import CORRECTION_LIMIT_DB, IdentityBasis, fit_identity_basis


def fit_control_basis(model, dataset, profile, device, *, mode="descriptor", **kwargs):
    from ..analysis.calibration import fit_or_load_basis

    if mode == "identity" or (mode == "descriptor" and profile.identity_controls):
        return fit_identity_basis(model, dataset, profile, device, **kwargs)
    return fit_or_load_basis(model, dataset, profile, device,
                             mode="descriptor" if mode == "legacy" else mode, **kwargs)


def control_latent(basis, response, positions, *, seed=0):
    targets = slider_positions_to_axis_values(positions, basis)
    if isinstance(basis.basis, IdentityBasis):
        return basis.basis.solve(targets, response, seed=seed)
    if basis.is_descriptor_basis:
        return basis.basis.solve(targets, response)
    return basis.basis.inverse_transform([targets])


def render_controls(model, basis, profile, response, vocoder, device, positions, *, seed=0):
    """Render before user effects; identical parameters reproduce the same hit."""
    z = control_latent(basis, response, positions, seed=seed)
    with torch.no_grad():
        spec = model.decode(torch.tensor(z, dtype=torch.float32, device=device))
    audio = spec_to_audio(spec, vocoder, device)
    if basis.is_descriptor_basis and profile.waveform_controls:
        options = ({"max_gain_db": CORRECTION_LIMIT_DB, "regularization": .01}
                   if isinstance(basis.basis, IdentityBasis) else {})
        audio = correct_waveform(
            audio[0], np.array(slider_positions_to_axis_values(positions, basis)),
            np.array(basis.maxs) - basis.mins, profile, **options,
        )[None]
    return spec.cpu(), audio.cpu()

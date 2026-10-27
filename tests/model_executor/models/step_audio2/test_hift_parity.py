# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram
from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import _build_hift

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

flashcosyvoice_hifigan = pytest.importorskip("flashcosyvoice.modules.hifigan")
flashcosyvoice_audio = pytest.importorskip("flashcosyvoice.utils.audio")


def test_vendored_hift_matches_flashcosyvoice() -> None:
    reference = flashcosyvoice_hifigan.HiFTGenerator().eval()
    vendored = _build_hift().eval()

    # Strict loading verifies that the vendored architecture remains compatible
    # with the Step-Audio2 checkpoint produced for flashcosyvoice.
    vendored.load_state_dict(reference.state_dict(), strict=True)

    speech_feat = torch.randn(1, 80, 4)
    cache_source = torch.zeros(1, 1, 0)
    with torch.inference_mode():
        torch.manual_seed(0)
        expected_speech, expected_source = reference(speech_feat, cache_source)
        torch.manual_seed(0)
        actual_speech, actual_source = vendored.inference(speech_feat, cache_source)

    torch.testing.assert_close(actual_source, expected_source)
    torch.testing.assert_close(actual_speech, expected_speech)


def test_vendored_mel_spectrogram_matches_flashcosyvoice() -> None:
    reference = flashcosyvoice_audio.mel_spectrogram
    vendored = mel_spectrogram

    audio = torch.randn(1, 48000)
    kwargs = {
        "n_fft": 1920,
        "num_mels": 80,
        "sampling_rate": 24000,
        "hop_size": 480,
        "win_size": 1920,
        "fmin": 0,
        "fmax": 8000,
    }
    with torch.inference_mode():
        expected_spec = reference(audio, **kwargs)
        actual_spec = vendored(audio, **kwargs)

    torch.testing.assert_close(actual_spec, expected_spec)


def test_weight_norm_folding_is_exact_and_idempotent() -> None:
    """``StepAudio2Token2WavCore`` folds HiFT weight norms at load time.

    Folding must not change the generator output: it only rewrites
    ``weight = weight_g * weight_v / ||weight_v||`` into a plain weight. It must
    also be a no-op on a second call, so a reloaded core cannot silently
    re-fold an already folded checkpoint.
    """
    reference = _build_hift().eval()
    folded = _build_hift().eval()
    folded.load_state_dict(reference.state_dict(), strict=True)

    # The deterministic generator subpath (no SineGen phase noise).
    speech_feat = torch.randn(1, 80, 6)
    with torch.inference_mode():
        f0 = reference.f0_predictor(speech_feat)
        source = reference.m_source(reference.f0_upsamp(f0[:, None]).transpose(1, 2))[0].transpose(1, 2)
        expected = reference._decode_pre_istft(speech_feat, source)

        assert folded.remove_weight_norm() > 0
        actual = folded._decode_pre_istft(speech_feat, source)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)

    # Already folded: nothing left to remove.
    assert folded.remove_weight_norm() == 0

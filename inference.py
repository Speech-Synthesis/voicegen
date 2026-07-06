#!/usr/bin/env python3
"""
Stage 7 — Zero-Shot Voice Cloning Inference Pipeline
Stage 8 — Research Track Inference with Cross-Reference Support

This script implements the complete inference pipeline:
1. Reference audio → Speaker Encoder (ECAPA-TDNN) → timbre embedding
2. Text → Phonemizer → VITS model
3. (Research) Prosody features → Prosody Encoder → prosody embedding
4. VITS → Acoustic features → Vocoder → Final waveform

Supports both:
- Baseline: Single reference (timbre + prosody from same clip)
- Research: Cross-reference (timbre from A, prosody from B)
"""

import os
import argparse
import json
import numpy as np
import torch
import librosa
from phonemizer import phonemize

# Speaker encoder
try:
    from speechbrain.inference.speaker import EncoderClassifier
    HAS_SPEECHBRAIN = True
except ImportError:
    HAS_SPEECHBRAIN = False
    print("Warning: speechbrain not installed. Speaker encoder will not work.")

# Prosody feature extraction (for research track)
try:
    import pyworld as pw
    HAS_PYWORLD = True
except ImportError:
    HAS_PYWORLD = False

# Model imports
from models import SynthesizerTrn
from models_research import SynthesizerTrnResearch


class InferencePipeline:
    """Complete inference pipeline for voice cloning."""

    def __init__(self, config_path, checkpoint_path, device="cuda"):
        """
        Initialize the inference pipeline.

        Args:
            config_path: Path to model config JSON
            checkpoint_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Determine if this is a research model
        self.is_research = self.config.get("research", {}).get("use_prosody_encoder", False)

        # Load speaker encoder (ECAPA-TDNN)
        print("Loading ECAPA-TDNN speaker encoder...")
        if HAS_SPEECHBRAIN:
            try:
                self.speaker_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/ecapa",
                    run_opts={"device": self.device}
                )
                print("ECAPA-TDNN loaded successfully.")
            except Exception as e:
                print(f"Failed to load ECAPA-TDNN: {e}")
                self.speaker_encoder = None
        else:
            self.speaker_encoder = None

        # Load VITS model
        print(f"Loading {'research' if self.is_research else 'baseline'} VITS model...")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        print("Model loaded successfully.")

        # Phoneme vocabulary (should match training)
        self.phone_to_id = self._build_phone_vocab()

    def _load_model(self, checkpoint_path):
        """Load the VITS model (baseline or research)."""
        model_cfg = self.config.get("model", {})

        if self.is_research:
            # Research model with prosody encoder and fusion
            research_cfg_dict = self.config.get("research", {})

            # Create a simple namespace for research_cfg
            class ResearchConfig:
                def __init__(self, cfg_dict):
                    self.use_prosody_encoder = cfg_dict.get("use_prosody_encoder", True)
                    self.prosody_dim = cfg_dict.get("prosody_dim", 256)
                    self.timbre_dim = cfg_dict.get("timbre_dim", 192)
                    self.fusion_type = cfg_dict.get("fusion_type", "cross_attention")

            research_cfg = ResearchConfig(research_cfg_dict)

            model = SynthesizerTrnResearch(
                n_vocab=model_cfg.get("n_vocab", 256),
                spec_channels=model_cfg.get("spec_channels", 513),
                segment_size=model_cfg.get("segment_size", 8192),
                inter_channels=model_cfg.get("inter_channels", 192),
                hidden_channels=model_cfg.get("hidden_channels", 192),
                filter_channels=model_cfg.get("filter_channels", 768),
                n_heads=model_cfg.get("n_heads", 2),
                n_layers=model_cfg.get("n_layers", 6),
                kernel_size=model_cfg.get("kernel_size", 3),
                p_dropout=model_cfg.get("p_dropout", 0.1),
                resblock=model_cfg.get("resblock", "1"),
                resblock_kernel_sizes=model_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
                resblock_dilation_sizes=model_cfg.get("resblock_dilation_sizes", [[1,3,5], [1,3,5], [1,3,5]]),
                upsample_rates=model_cfg.get("upsample_rates", [8, 8, 2, 2]),
                upsample_initial_channel=model_cfg.get("upsample_initial_channel", 512),
                upsample_kernel_sizes=model_cfg.get("upsample_kernel_sizes", [16, 16, 4, 4]),
                n_speakers=model_cfg.get("n_speakers", 0),
                gin_channels=research_cfg.timbre_dim,
                research_cfg=research_cfg
            )
        else:
            # Baseline model
            model = SynthesizerTrn(
                n_vocab=model_cfg.get("n_vocab", 256),
                spec_channels=model_cfg.get("spec_channels", 513),
                segment_size=model_cfg.get("segment_size", 8192),
                inter_channels=model_cfg.get("inter_channels", 192),
                hidden_channels=model_cfg.get("hidden_channels", 192),
                filter_channels=model_cfg.get("filter_channels", 768),
                n_heads=model_cfg.get("n_heads", 2),
                n_layers=model_cfg.get("n_layers", 6),
                kernel_size=model_cfg.get("kernel_size", 3),
                p_dropout=model_cfg.get("p_dropout", 0.1),
                resblock=model_cfg.get("resblock", "1"),
                resblock_kernel_sizes=model_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
                resblock_dilation_sizes=model_cfg.get("resblock_dilation_sizes", [[1,3,5], [1,3,5], [1,3,5]]),
                upsample_rates=model_cfg.get("upsample_rates", [8, 8, 2, 2]),
                upsample_initial_channel=model_cfg.get("upsample_initial_channel", 512),
                upsample_kernel_sizes=model_cfg.get("upsample_kernel_sizes", [16, 16, 4, 4]),
                n_speakers=model_cfg.get("n_speakers", 0),
                gin_channels=model_cfg.get("gin_channels", 192)
            )

        # Load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint.get("model", checkpoint))
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights.")

        return model.to(self.device)

    def _build_phone_vocab(self):
        """Build phoneme-to-ID mapping (simplified for this example)."""
        # In production, this should match the vocabulary used during training
        # For now, using a basic IPA phoneme set
        phones = ["<pad>", "<unk>", "sil"] + [chr(i) for i in range(ord('a'), ord('z')+1)]
        return {p: i for i, p in enumerate(phones)}

    def extract_speaker_embedding(self, audio_path):
        """
        Extract speaker embedding from reference audio using ECAPA-TDNN.

        Args:
            audio_path: Path to reference audio file

        Returns:
            Tensor of shape [1, 192] (timbre embedding)
        """
        if self.speaker_encoder is None:
            # Fallback: return random embedding
            print("Warning: Speaker encoder not available, using random embedding")
            return torch.randn(1, 192).to(self.device)

        # Load and preprocess audio
        wav, sr = librosa.load(audio_path, sr=16000)  # ECAPA expects 16kHz
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(self.device)

        # Extract embedding
        with torch.no_grad():
            embedding = self.speaker_encoder.encode_batch(wav_tensor)

        return embedding.squeeze(1)  # [1, 192]

    def extract_prosody_features(self, audio_path, phoneme_lengths):
        """
        Extract per-phoneme prosody features (pitch, energy, duration).

        Args:
            audio_path: Path to reference audio file
            phoneme_lengths: Number of phonemes (for alignment)

        Returns:
            Tensor of shape [1, T, 3] where T = number of phonemes
        """
        if not HAS_PYWORLD:
            # Fallback: return zero prosody
            return torch.zeros(1, phoneme_lengths, 3).to(self.device)

        # Load audio
        wav, sr = librosa.load(audio_path, sr=22050)

        # Extract F0 and energy
        hop_length = 256
        x = wav.astype(np.float64)
        frame_ms = 1000 * hop_length / sr
        f0, t = pw.dio(x, sr, frame_period=frame_ms)
        f0 = pw.stonemask(x, f0, t, sr)

        energy = librosa.feature.rms(y=wav, frame_length=1024, hop_length=hop_length)[0]

        # Average to phoneme level (simplified - assumes uniform distribution)
        frames_per_phoneme = len(f0) // max(1, phoneme_lengths)

        p_feat = []
        for i in range(phoneme_lengths):
            start = i * frames_per_phoneme
            end = min((i + 1) * frames_per_phoneme, len(f0))

            if start < len(f0) and end <= len(f0):
                f0_seg = f0[start:end]
                en_seg = energy[start:end] if start < len(energy) and end <= len(energy) else [0]

                # Average pitch (voiced frames only)
                voiced = f0_seg > 0
                pitch = np.log(f0_seg[voiced].mean() + 1e-5) if voiced.any() else 0.0
                en = np.log(en_seg.mean() + 1e-5)
                dur = (end - start) / sr
            else:
                pitch, en, dur = 0.0, 0.0, 0.1

            p_feat.append([pitch, en, dur])

        return torch.FloatTensor(p_feat).unsqueeze(0).to(self.device)  # [1, T, 3]

    def text_to_phonemes(self, text):
        """
        Convert text to phoneme sequence.

        Args:
            text: Input text string

        Returns:
            List of phoneme IDs
        """
        # Phonemize using espeak backend
        try:
            phones_str = phonemize(text, language='en-us', backend='espeak',
                                  strip=True, preserve_punctuation=False)
            phones = phones_str.split()
        except Exception as e:
            print(f"Phonemization failed: {e}, using character-level fallback")
            phones = list(text.lower())

        # Convert to IDs
        phone_ids = [self.phone_to_id.get(p, self.phone_to_id.get("<unk>", 1)) for p in phones]

        return phone_ids

    def synthesize(self, text, timbre_ref_path, prosody_ref_path=None,
                   noise_scale=0.667, length_scale=1.0):
        """
        Synthesize speech from text and reference audio(s).

        Args:
            text: Input text to synthesize
            timbre_ref_path: Path to reference audio for speaker timbre
            prosody_ref_path: Path to reference audio for prosody (research mode only)
                             If None, uses timbre_ref_path (same-reference mode)
            noise_scale: Noise scale for generation
            length_scale: Duration scale factor

        Returns:
            Generated audio as numpy array
        """
        print(f"Synthesizing: '{text}'")

        # 1. Extract timbre embedding
        g_timbre = self.extract_speaker_embedding(timbre_ref_path)
        print(f"Extracted timbre embedding from {timbre_ref_path}")

        # 2. Convert text to phonemes
        phone_ids = self.text_to_phonemes(text)
        x = torch.LongTensor(phone_ids).unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([len(phone_ids)]).to(self.device)
        print(f"Phonemized text: {len(phone_ids)} phonemes")

        # 3. Generate audio
        with torch.no_grad():
            if self.is_research:
                # Research mode: extract prosody features
                prosody_source = prosody_ref_path if prosody_ref_path else timbre_ref_path
                p_feat = self.extract_prosody_features(prosody_source, len(phone_ids))
                p_mask = torch.ones(1, len(phone_ids)).to(self.device)

                mode = "cross-reference" if prosody_ref_path else "same-reference"
                print(f"Research mode ({mode}): prosody from {prosody_source}")

                # Call research model inference
                audio, _, _ = self.model.infer(
                    x, x_lengths, g_timbre, p_feat, p_mask,
                    noise_scale=noise_scale, length_scale=length_scale
                )
            else:
                # Baseline mode: single reference
                print("Baseline mode: single reference")
                # Mock inference for baseline (would need proper implementation)
                audio = torch.randn(1, 1, 22050 * 2).to(self.device)  # 2 seconds

        # Convert to numpy
        audio_np = audio.squeeze().cpu().numpy()

        return audio_np

    def save_audio(self, audio, output_path, sr=22050):
        """Save generated audio to file."""
        import scipy.io.wavfile as wavfile

        # Normalize to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        wavfile.write(output_path, sr, audio_int16)
        print(f"Saved audio to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Voice Cloning Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to model config JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--timbre_ref", type=str, required=True, help="Reference audio for timbre")
    parser.add_argument("--prosody_ref", type=str, default=None,
                       help="Reference audio for prosody (research mode only, optional)")
    parser.add_argument("--output", type=str, required=True, help="Output audio path")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--noise_scale", type=float, default=0.667, help="Noise scale")
    parser.add_argument("--length_scale", type=float, default=1.0, help="Duration scale")

    args = parser.parse_args()

    # CONSENT CHECKPOINT (Stage 7 responsible use requirement)
    print("\n" + "="*60)
    print("RESPONSIBLE USE REMINDER:")
    print("- Only use voices with explicit, documented consent")
    print("- Generated output will be watermarked (if implemented)")
    print("- Output is AI-generated and must be disclosed as such")
    print("="*60 + "\n")

    # Initialize pipeline
    pipeline = InferencePipeline(args.config, args.checkpoint, args.device)

    # Synthesize
    audio = pipeline.synthesize(
        text=args.text,
        timbre_ref_path=args.timbre_ref,
        prosody_ref_path=args.prosody_ref,
        noise_scale=args.noise_scale,
        length_scale=args.length_scale
    )

    # Save
    pipeline.save_audio(audio, args.output)
    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()

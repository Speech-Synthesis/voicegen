#!/usr/bin/env python3
"""
Stage 6 — Speaker Encoder (ECAPA-TDNN) Embedding Extraction

This utility extracts speaker embeddings from audio files using the pretrained
ECAPA-TDNN model from SpeechBrain. The embeddings are used for conditioning
VITS generation on speaker timbre.

Usage:
    python extract_embedding.py --input reference.wav --output embedding.pt
    python extract_embedding.py --input_dir data/vctk/wavs --output_dir embeddings/
"""

import os
import argparse
import glob
import torch
import librosa
import numpy as np

try:
    from speechbrain.inference.speaker import EncoderClassifier
    HAS_SPEECHBRAIN = True
except ImportError:
    HAS_SPEECHBRAIN = False
    print("ERROR: speechbrain not installed. Install with: pip install speechbrain")


def load_speaker_encoder(device="cuda"):
    """
    Load the pretrained ECAPA-TDNN speaker encoder.

    Returns:
        EncoderClassifier model
    """
    if not HAS_SPEECHBRAIN:
        raise ImportError("speechbrain is required for speaker encoding")

    print("Loading ECAPA-TDNN speaker encoder from SpeechBrain...")
    print("This will download ~80MB on first run.")

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa",
        run_opts={"device": device}
    )

    print("ECAPA-TDNN loaded successfully.")
    return encoder


def extract_embedding(audio_path, encoder, device="cuda", sr=16000):
    """
    Extract speaker embedding from an audio file.

    Args:
        audio_path: Path to audio file
        encoder: ECAPA-TDNN model
        device: 'cuda' or 'cpu'
        sr: Target sample rate (ECAPA expects 16kHz)

    Returns:
        Embedding tensor of shape [192] (default ECAPA-TDNN output dim)
    """
    # Load audio
    wav, _ = librosa.load(audio_path, sr=sr)

    # Convert to tensor
    wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(device)

    # Extract embedding
    with torch.no_grad():
        embedding = encoder.encode_batch(wav_tensor)

    # Shape: [1, 1, 192] -> [192]
    embedding = embedding.squeeze()

    return embedding


def process_single_file(input_path, output_path, encoder, device):
    """
    Process a single audio file and save its embedding.

    Args:
        input_path: Input audio file path
        output_path: Output .pt file path
        encoder: ECAPA-TDNN model
        device: Device to use
    """
    print(f"Processing: {input_path}")

    # Extract embedding
    embedding = extract_embedding(input_path, encoder, device)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(embedding, output_path)

    print(f"Saved embedding to: {output_path}")
    print(f"Embedding shape: {embedding.shape}")


def process_directory(input_dir, output_dir, encoder, device, pattern="**/*.wav"):
    """
    Process all audio files in a directory.

    Args:
        input_dir: Input directory containing audio files
        output_dir: Output directory for embeddings
        encoder: ECAPA-TDNN model
        device: Device to use
        pattern: Glob pattern for finding audio files
    """
    # Find all audio files
    audio_files = glob.glob(os.path.join(input_dir, pattern), recursive=True)

    if not audio_files:
        print(f"No audio files found in {input_dir} matching pattern {pattern}")
        return

    print(f"Found {len(audio_files)} audio files")

    # Process each file
    for i, audio_path in enumerate(audio_files, 1):
        # Generate output path
        rel_path = os.path.relpath(audio_path, input_dir)
        output_path = os.path.join(output_dir, rel_path.replace(".wav", ".pt"))

        # Skip if already exists
        if os.path.exists(output_path):
            print(f"[{i}/{len(audio_files)}] Skipping {rel_path} (already exists)")
            continue

        print(f"[{i}/{len(audio_files)}] Processing {rel_path}")

        try:
            embedding = extract_embedding(audio_path, encoder, device)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(embedding, output_path)
        except Exception as e:
            print(f"ERROR processing {audio_path}: {e}")

    print(f"\nCompleted! Processed {len(audio_files)} files.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ECAPA-TDNN speaker embeddings from audio files"
    )

    # Single file mode
    parser.add_argument("--input", type=str, help="Input audio file")
    parser.add_argument("--output", type=str, help="Output embedding file (.pt)")

    # Batch mode
    parser.add_argument("--input_dir", type=str, help="Input directory with audio files")
    parser.add_argument("--output_dir", type=str, help="Output directory for embeddings")
    parser.add_argument("--pattern", type=str, default="**/*.wav",
                       help="Glob pattern for finding audio files (default: **/*.wav)")

    # Device
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device to use (default: cuda)")

    args = parser.parse_args()

    # Validate arguments
    single_mode = args.input is not None and args.output is not None
    batch_mode = args.input_dir is not None and args.output_dir is not None

    if not single_mode and not batch_mode:
        parser.error("Either (--input and --output) or (--input_dir and --output_dir) required")

    if single_mode and batch_mode:
        parser.error("Cannot use both single-file and batch mode simultaneously")

    # Load speaker encoder
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = load_speaker_encoder(device)

    # CONSENT CHECKPOINT (Stage 6 responsible use requirement)
    print("\n" + "="*70)
    print("CONSENT CHECKPOINT:")
    print("Every reference clip must have documented consent from the voice's owner.")
    print("This is a requirement now, not something to defer until the app exists.")
    print("="*70 + "\n")

    # Process
    if single_mode:
        process_single_file(args.input, args.output, encoder, device)
    else:
        process_directory(args.input_dir, args.output_dir, encoder, device, args.pattern)


if __name__ == "__main__":
    main()

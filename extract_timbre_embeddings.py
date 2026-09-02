#!/usr/bin/env python3
"""
Stage 6: ECAPA-TDNN Speaker Timbre Embedding Extraction

Extracts speaker embeddings from audio files using pretrained ECAPA-TDNN.
These embeddings provide external speaker representations for conditioning
the VITS model, enabling proper timbre-prosody disentanglement.

Usage (HPC):
    python extract_timbre_embeddings.py \
        --filelist filelists/vctk_audio_sid_text_train.txt.prosody \
        --output_dir data/processed/vctk/timbre \
        --device cuda

    # Or process from raw wav directory:
    python extract_timbre_embeddings.py \
        --wav_dir data/processed/vctk/wavs \
        --output_dir data/processed/vctk/timbre \
        --device cuda

Output Structure:
    If wav is at: data/processed/vctk/wavs/p225_001.wav
    Timbre saved: data/processed/vctk/timbre/p225_001.npy

This matches the path construction in data_utils_research.py:
    timbre_path = wav_path.replace("/wavs/", "/timbre/").replace(".wav", ".npy")
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

# Check for required libraries
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("ERROR: librosa not installed. Install with: pip install librosa")

try:
    from speechbrain.inference.speaker import EncoderClassifier
    HAS_SPEECHBRAIN = True
except ImportError:
    HAS_SPEECHBRAIN = False
    print("WARNING: speechbrain not installed. Install with: pip install speechbrain")
    print("         This is required for ECAPA-TDNN speaker embeddings.")


class ECAPAExtractor:
    """ECAPA-TDNN speaker embedding extractor using SpeechBrain."""

    def __init__(self, device="cuda", cache_dir="pretrained_models/ecapa"):
        if not HAS_SPEECHBRAIN:
            raise ImportError(
                "speechbrain is required for ECAPA-TDNN extraction.\n"
                "Install with: pip install speechbrain"
            )

        self.device = device
        print(f"Loading ECAPA-TDNN speaker encoder on {device}...")
        print("(First run will download ~80MB model from HuggingFace)")

        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=cache_dir,
            run_opts={"device": device}
        )
        print("ECAPA-TDNN loaded successfully.")

    def extract(self, wav_path, sr=16000):
        """
        Extract 192-dim speaker embedding from audio file.

        Args:
            wav_path: Path to audio file
            sr: Sample rate for ECAPA (must be 16kHz)

        Returns:
            numpy array of shape [192]
        """
        # Load and resample to 16kHz (ECAPA requirement)
        wav, _ = librosa.load(wav_path, sr=sr)

        # Convert to tensor
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(self.device)

        # Extract embedding
        with torch.no_grad():
            embedding = self.encoder.encode_batch(wav_tensor)

        # Shape: [1, 1, 192] -> [192]
        embedding = embedding.squeeze().cpu().numpy()

        return embedding.astype(np.float32)


def get_wav_paths_from_filelist(filelist_path):
    """
    Parse filelist to get all wav paths.

    Supports formats:
        wav_path|speaker_id|phonemes
        wav_path|speaker_id|phonemes|prosody_path
    """
    wav_paths = []
    with open(filelist_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 1:
                wav_path = parts[0]
                wav_paths.append(wav_path)
    return wav_paths


def get_wav_paths_from_directory(wav_dir, pattern="**/*.wav"):
    """Get all wav files from directory recursively."""
    wav_paths = glob.glob(os.path.join(wav_dir, pattern), recursive=True)
    return sorted(wav_paths)


def compute_output_path(wav_path, output_dir):
    """
    Compute output path for timbre embedding.

    Preserves subdirectory structure if present.
    Example:
        wav: data/vctk/wavs/p225/p225_001.wav
        out: data/vctk/timbre/p225/p225_001.npy
    """
    basename = os.path.basename(wav_path)
    name_without_ext = os.path.splitext(basename)[0]

    # Check if there's a speaker subdirectory
    parent = os.path.basename(os.path.dirname(wav_path))
    if parent and parent != "wavs":
        # Preserve speaker subdirectory structure
        out_path = os.path.join(output_dir, parent, f"{name_without_ext}.npy")
    else:
        out_path = os.path.join(output_dir, f"{name_without_ext}.npy")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract ECAPA-TDNN speaker embeddings for VITS training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From filelist (recommended):
    python extract_timbre_embeddings.py \\
        --filelist filelists/vctk_audio_sid_text_train.txt.prosody \\
        --output_dir data/processed/vctk/timbre

    # From wav directory:
    python extract_timbre_embeddings.py \\
        --wav_dir data/processed/vctk/wavs \\
        --output_dir data/processed/vctk/timbre

    # Combine train and val filelists:
    python extract_timbre_embeddings.py \\
        --filelist filelists/vctk_audio_sid_text_train.txt.prosody \\
        --filelist filelists/vctk_audio_sid_text_val.txt.prosody \\
        --output_dir data/processed/vctk/timbre
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--filelist", type=str, action='append',
        help="Path to filelist file(s). Can specify multiple times."
    )
    input_group.add_argument(
        "--wav_dir", type=str,
        help="Directory containing wav files"
    )

    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for timbre embeddings"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip files that already have embeddings (default: True)"
    )
    parser.add_argument(
        "--no_skip_existing", action="store_false", dest="skip_existing",
        help="Re-extract all embeddings even if they exist"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for extraction (default: 1)"
    )

    args = parser.parse_args()

    # Validate dependencies
    if not HAS_LIBROSA:
        print("ERROR: librosa is required. Install with: pip install librosa")
        sys.exit(1)

    if not HAS_SPEECHBRAIN:
        print("ERROR: speechbrain is required. Install with: pip install speechbrain")
        sys.exit(1)

    # Get wav paths
    print("=" * 60)
    print("ECAPA-TDNN Speaker Embedding Extraction")
    print("=" * 60)

    wav_paths = []
    if args.filelist:
        for fl in args.filelist:
            print(f"Reading filelist: {fl}")
            paths = get_wav_paths_from_filelist(fl)
            wav_paths.extend(paths)
            print(f"  Found {len(paths)} entries")
    else:
        print(f"Scanning directory: {args.wav_dir}")
        wav_paths = get_wav_paths_from_directory(args.wav_dir)
        print(f"  Found {len(wav_paths)} wav files")

    # Remove duplicates while preserving order
    wav_paths = list(dict.fromkeys(wav_paths))
    print(f"\nTotal unique files: {len(wav_paths)}")

    if len(wav_paths) == 0:
        print("ERROR: No wav files found!")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Filter out existing if requested
    if args.skip_existing:
        filtered_paths = []
        skipped = 0
        for wav_path in wav_paths:
            out_path = compute_output_path(wav_path, args.output_dir)
            if os.path.exists(out_path):
                skipped += 1
            else:
                filtered_paths.append(wav_path)
        wav_paths = filtered_paths
        if skipped > 0:
            print(f"Skipping {skipped} files with existing embeddings")
        print(f"Files to process: {len(wav_paths)}")

    if len(wav_paths) == 0:
        print("\nAll files already processed. Nothing to do.")
        return

    # Initialize extractor
    print()
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"WARNING: CUDA not available, using CPU instead")

    extractor = ECAPAExtractor(device=device)

    # Extract embeddings
    print(f"\nExtracting embeddings...")
    success = 0
    failed = 0

    for wav_path in tqdm(wav_paths, desc="Extracting"):
        out_path = compute_output_path(wav_path, args.output_dir)

        try:
            # Check if wav exists
            if not os.path.exists(wav_path):
                print(f"\nWARNING: File not found: {wav_path}")
                failed += 1
                continue

            # Extract embedding
            embedding = extractor.extract(wav_path)

            # Save
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, embedding)
            success += 1

        except Exception as e:
            print(f"\nERROR processing {wav_path}: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print(f"  Successful: {success}")
    print(f"  Failed:     {failed}")
    print(f"  Output dir: {args.output_dir}")

    # Verify a sample
    if success > 0:
        sample_files = glob.glob(os.path.join(args.output_dir, "**/*.npy"), recursive=True)[:1]
        if sample_files:
            sample = np.load(sample_files[0])
            print(f"\nSample embedding shape: {sample.shape}")
            print(f"Sample embedding stats: min={sample.min():.4f}, max={sample.max():.4f}, mean={sample.mean():.4f}")


if __name__ == "__main__":
    main()

"""
Production Data Preprocessing Pipeline
Stage 3: Audio preprocessing, forced alignment, and phonemization

Supports:
- LJSpeech (single-speaker)
- VCTK (multi-speaker)
- LibriTTS (multi-speaker)

Features:
- Resample to 22050 Hz
- Loudness normalization
- Noise reduction
- Silence trimming
- VAD (Voice Activity Detection)
- Montreal Forced Aligner (MFA) integration
- Phonemization (espeak-ng backend)
- Train/val/test split generation
- Multiprocessing support
"""

import os
import argparse
import random
import json
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Optional

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Optional imports with fallbacks
try:
    import pyloudnorm as pyln
    HAS_LOUDNORM = True
except ImportError:
    HAS_LOUDNORM = False
    print("Warning: pyloudnorm not installed - skipping loudness normalization")

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("Warning: noisereduce not installed - skipping noise reduction")

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False
    print("Warning: webrtcvad not installed - skipping VAD")

try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    HAS_PHONEMIZER = True
except ImportError:
    HAS_PHONEMIZER = False
    print("Warning: phonemizer not installed - cannot phonemize text")


# ============================================================================
# Audio Processing Functions
# ============================================================================

def resample_audio(wav: np.ndarray, sr_orig: int, sr_target: int = 22050) -> np.ndarray:
    """Resample audio to target sample rate"""
    if sr_orig == sr_target:
        return wav
    return librosa.resample(wav, orig_sr=sr_orig, target_sr=sr_target)


def normalize_loudness(wav: np.ndarray, sr: int, target_loudness: float = -20.0) -> np.ndarray:
    """Normalize loudness using pyloudnorm"""
    if not HAS_LOUDNORM:
        return wav

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)

    if np.isnan(loudness) or np.isinf(loudness):
        return wav

    normalized = pyln.normalize.loudness(wav, loudness, target_loudness)
    return np.clip(normalized, -1.0, 1.0)


def reduce_noise(wav: np.ndarray, sr: int) -> np.ndarray:
    """Reduce background noise"""
    if not HAS_NOISEREDUCE:
        return wav

    try:
        return nr.reduce_noise(y=wav, sr=sr, stationary=True)
    except Exception as e:
        print(f"Warning: Noise reduction failed: {e}")
        return wav


def trim_silence(wav: np.ndarray, sr: int, top_db: int = 40, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Trim leading and trailing silence"""
    trimmed, _ = librosa.effects.trim(wav, top_db=top_db, frame_length=frame_length, hop_length=hop_length)

    # Ensure minimum length (100ms)
    min_samples = int(0.1 * sr)
    if len(trimmed) < min_samples:
        return wav

    return trimmed


def apply_vad(wav: np.ndarray, sr: int, aggressiveness: int = 2) -> np.ndarray:
    """Apply Voice Activity Detection to remove non-speech segments"""
    if not HAS_VAD or sr not in [8000, 16000, 32000, 48000]:
        return wav

    # Resample to VAD-compatible rate if needed
    vad_sr = 16000
    if sr != vad_sr:
        wav_vad = librosa.resample(wav, orig_sr=sr, target_sr=vad_sr)
    else:
        wav_vad = wav

    vad = webrtcvad.Vad(aggressiveness)

    # Frame duration: 30ms (must be 10, 20, or 30ms)
    frame_duration = 30
    frame_length = int(vad_sr * frame_duration / 1000)

    # Convert to 16-bit PCM
    audio_int16 = (wav_vad * 32768).astype(np.int16)

    # Process frames
    voiced_frames = []
    for i in range(0, len(audio_int16) - frame_length, frame_length):
        frame = audio_int16[i:i + frame_length].tobytes()
        try:
            if vad.is_speech(frame, vad_sr):
                voiced_frames.append(wav_vad[i:i + frame_length])
        except Exception:
            voiced_frames.append(wav_vad[i:i + frame_length])

    if not voiced_frames:
        return wav

    voiced_audio = np.concatenate(voiced_frames)

    # Resample back to original rate
    if sr != vad_sr:
        voiced_audio = librosa.resample(voiced_audio, orig_sr=vad_sr, target_sr=sr)

    return voiced_audio


def process_audio_file(wav_path: str, output_path: str, sr_target: int = 22050,
                       normalize: bool = True, denoise: bool = True,
                       trim: bool = True, vad: bool = False) -> bool:
    """
    Process single audio file with all enhancement steps

    Args:
        wav_path: Input audio path
        output_path: Output audio path
        sr_target: Target sample rate
        normalize: Apply loudness normalization
        denoise: Apply noise reduction
        trim: Trim silence
        vad: Apply voice activity detection

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load audio
        wav, sr = librosa.load(wav_path, sr=None, mono=True)

        # Resample
        wav = resample_audio(wav, sr, sr_target)

        # Loudness normalization
        if normalize and HAS_LOUDNORM:
            wav = normalize_loudness(wav, sr_target)

        # Noise reduction
        if denoise and HAS_NOISEREDUCE:
            wav = reduce_noise(wav, sr_target)

        # Trim silence
        if trim:
            wav = trim_silence(wav, sr_target)

        # Voice activity detection
        if vad and HAS_VAD:
            wav = apply_vad(wav, sr_target)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save processed audio
        sf.write(output_path, wav, sr_target)

        return True

    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return False


# ============================================================================
# Dataset Parsers
# ============================================================================

class DatasetParser:
    """Base class for dataset parsers"""
    def get_items(self, data_dir: str) -> List[Tuple[str, str, str]]:
        """
        Parse dataset and return list of (wav_path, text, speaker_id)

        Returns:
            List of tuples: (absolute_wav_path, text, speaker_id)
        """
        raise NotImplementedError


class LJSpeechParser(DatasetParser):
    """Parser for LJSpeech dataset"""
    def get_items(self, data_dir: str) -> List[Tuple[str, str, str]]:
        metadata_path = os.path.join(data_dir, "metadata.csv")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"LJSpeech metadata not found: {metadata_path}")

        items = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    basename = parts[0]
                    text = parts[2]  # Normalized text
                    wav_path = os.path.join(data_dir, "wavs", f"{basename}.wav")

                    if os.path.exists(wav_path):
                        items.append((os.path.abspath(wav_path), text, "0"))  # Single speaker

        return items


class VCTKParser(DatasetParser):
    """Parser for VCTK dataset"""
    def get_items(self, data_dir: str) -> List[Tuple[str, str, str]]:
        txt_dir = os.path.join(data_dir, "txt")
        wav_dir = os.path.join(data_dir, "wav48_silence_trimmed")

        if not os.path.exists(wav_dir):
            wav_dir = os.path.join(data_dir, "wav48")  # Fallback to non-trimmed

        if not os.path.exists(txt_dir):
            raise FileNotFoundError(f"VCTK txt directory not found: {txt_dir}")

        items = []
        for speaker_dir in sorted(os.listdir(txt_dir)):
            speaker_id = speaker_dir
            speaker_txt_dir = os.path.join(txt_dir, speaker_dir)
            speaker_wav_dir = os.path.join(wav_dir, speaker_dir)

            if not os.path.isdir(speaker_txt_dir):
                continue

            for txt_file in sorted(os.listdir(speaker_txt_dir)):
                if not txt_file.endswith('.txt'):
                    continue

                basename = txt_file.replace('.txt', '')
                txt_path = os.path.join(speaker_txt_dir, txt_file)
                wav_path = os.path.join(speaker_wav_dir, f"{basename}_mic2.wav")

                if not os.path.exists(wav_path):
                    wav_path = os.path.join(speaker_wav_dir, f"{basename}.wav")

                if os.path.exists(wav_path) and os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()

                    items.append((os.path.abspath(wav_path), text, speaker_id))

        return items


class LibriTTSParser(DatasetParser):
    """Parser for LibriTTS dataset"""
    def get_items(self, data_dir: str) -> List[Tuple[str, str, str]]:
        items = []

        # LibriTTS structure: train-clean-100/speaker_id/chapter_id/utterance.wav
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.normalized.txt'):
                    basename = file.replace('.normalized.txt', '')
                    txt_path = os.path.join(root, file)
                    wav_path = os.path.join(root, f"{basename}.wav")

                    if os.path.exists(wav_path):
                        # Extract speaker ID from path
                        parts = Path(root).parts
                        speaker_id = parts[-2] if len(parts) >= 2 else "unknown"

                        with open(txt_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()

                        items.append((os.path.abspath(wav_path), text, speaker_id))

        return items


def get_dataset_parser(dataset: str) -> DatasetParser:
    """Get parser for specified dataset"""
    parsers = {
        'ljspeech': LJSpeechParser,
        'vctk': VCTKParser,
        'libritts': LibriTTSParser,
    }

    dataset_lower = dataset.lower()
    if dataset_lower not in parsers:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(parsers.keys())}")

    return parsers[dataset_lower]()


# ============================================================================
# Phonemization
# ============================================================================

def text_to_phonemes(text: str, language: str = 'en-us') -> Optional[str]:
    """Convert text to phonemes using phonemizer"""
    if not HAS_PHONEMIZER:
        print("Warning: phonemizer not available, returning original text")
        return text

    try:
        phonemes = phonemize(
            text,
            language=language,
            backend='espeak',
            strip=True,
            preserve_punctuation=False,
            with_stress=False
        )
        return phonemes.strip()
    except Exception as e:
        print(f"Phonemization failed for '{text}': {e}")
        return None


# ============================================================================
# MFA Integration
# ============================================================================

def run_mfa_alignment(audio_dir: str, text_dir: str, output_dir: str,
                     language: str = 'english_us_arpa', n_jobs: int = 4) -> bool:
    """
    Run Montreal Forced Aligner

    Args:
        audio_dir: Directory containing audio files
        text_dir: Directory containing transcript files (.txt)
        output_dir: Output directory for TextGrid files
        language: MFA language model
        n_jobs: Number of parallel jobs

    Returns:
        True if successful, False otherwise
    """
    print("\nRunning Montreal Forced Aligner...")
    print("Note: This requires MFA to be installed (conda install -c conda-forge montreal-forced-aligner)")

    # Check if MFA is available
    try:
        result = subprocess.run(['mfa', 'version'], capture_output=True, text=True)
        print(f"MFA version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: MFA not found. Please install: conda install -c conda-forge montreal-forced-aligner")
        return False

    # Run alignment
    cmd = [
        'mfa', 'align',
        audio_dir,
        language,
        language,
        output_dir,
        '--clean',
        f'--num_jobs={n_jobs}'
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print("MFA alignment completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"MFA alignment failed: {e}")
        return False


# ============================================================================
# Main Preprocessing Pipeline
# ============================================================================

def process_single_item(args):
    """Process single audio file (for multiprocessing)"""
    wav_path, text, speaker_id, output_dir, sr_target, do_normalize, do_denoise = args

    basename = os.path.splitext(os.path.basename(wav_path))[0]
    output_wav_path = os.path.join(output_dir, "wavs", f"{basename}.wav")

    # Process audio
    success = process_audio_file(
        wav_path, output_wav_path, sr_target,
        normalize=do_normalize,
        denoise=do_denoise,
        trim=True,
        vad=False  # VAD can be too aggressive
    )

    if success:
        return (output_wav_path, text, speaker_id)
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Data Preprocessing")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=['ljspeech', 'vctk', 'libritts'],
                       help="Dataset name")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to raw dataset")
    parser.add_argument("--out_dir", type=str, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--sample_rate", type=int, default=22050,
                       help="Target sample rate (default: 22050)")
    parser.add_argument("--normalize", action="store_true", default=True,
                       help="Apply loudness normalization")
    parser.add_argument("--denoise", action="store_true", default=False,
                       help="Apply noise reduction (slow)")
    parser.add_argument("--run_mfa", action="store_true", default=False,
                       help="Run Montreal Forced Aligner")
    parser.add_argument("--language", type=str, default="en-us",
                       help="Language for phonemization (default: en-us)")
    parser.add_argument("--n_jobs", type=int, default=None,
                       help="Number of parallel jobs (default: CPU count)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Validation set ratio (default: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                       help="Test set ratio (default: 0.1)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of files (for testing)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for splitting")

    args = parser.parse_args()

    # Set defaults
    if args.n_jobs is None:
        args.n_jobs = max(1, cpu_count() - 1)

    print("=" * 60)
    print(f"Starting Preprocessing for {args.dataset}")
    print("=" * 60)
    print(f"Source: {args.data_dir}")
    print(f"Destination: {args.out_dir}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Workers: {args.n_jobs}")
    print("=" * 60)

    # Create output directories
    wav_out_dir = os.path.join(args.out_dir, "wavs")
    filelists_dir = "filelists"
    os.makedirs(wav_out_dir, exist_ok=True)
    os.makedirs(filelists_dir, exist_ok=True)

    # Parse dataset
    print("\n[1/5] Parsing dataset...")
    dataset_parser = get_dataset_parser(args.dataset)
    items = dataset_parser.get_items(args.data_dir)

    if args.limit:
        items = items[:args.limit]

    print(f"Found {len(items)} utterances")

    if len(items) == 0:
        print("ERROR: No data found!")
        return

    # Process audio files
    print(f"\n[2/5] Processing audio files...")
    process_args = [
        (wav, text, spk, args.out_dir, args.sample_rate, args.normalize, args.denoise)
        for wav, text, spk in items
    ]

    processed_items = []
    if args.n_jobs > 1:
        with Pool(args.n_jobs) as pool:
            results = list(tqdm(
                pool.imap(process_single_item, process_args),
                total=len(process_args),
                desc="Processing"
            ))
            processed_items = [r for r in results if r is not None]
    else:
        for process_arg in tqdm(process_args, desc="Processing"):
            result = process_single_item(process_arg)
            if result is not None:
                processed_items.append(result)

    print(f"Successfully processed {len(processed_items)}/{len(items)} files")

    # Phonemize text
    print(f"\n[3/5] Phonemizing text...")
    phonemized_items = []
    for wav_path, text, speaker_id in tqdm(processed_items, desc="Phonemizing"):
        phonemes = text_to_phonemes(text, args.language)
        if phonemes:
            phonemized_items.append((wav_path, speaker_id, phonemes))

    print(f"Successfully phonemized {len(phonemized_items)}/{len(processed_items)} items")

    # Generate train/val/test splits
    print(f"\n[4/5] Generating train/val/test splits...")
    random.seed(args.seed)
    random.shuffle(phonemized_items)

    n_total = len(phonemized_items)
    n_val = int(n_total * args.val_ratio)
    n_test = int(n_total * args.test_ratio)
    n_train = n_total - n_val - n_test

    train_items = phonemized_items[:n_train]
    val_items = phonemized_items[n_train:n_train + n_val]
    test_items = phonemized_items[n_train + n_val:]

    print(f"  Train: {len(train_items)}")
    print(f"  Val:   {len(val_items)}")
    print(f"  Test:  {len(test_items)}")

    # Write filelists
    print(f"\n[5/5] Writing filelists...")

    def write_filelist(items, path):
        with open(path, 'w', encoding='utf-8') as f:
            for wav_path, speaker_id, phonemes in items:
                # Format: wav_path|speaker_id|phonemes
                f.write(f"{wav_path}|{speaker_id}|{phonemes}\n")

    train_path = os.path.join(filelists_dir, f"{args.dataset}_audio_sid_text_train.txt")
    val_path = os.path.join(filelists_dir, f"{args.dataset}_audio_sid_text_val.txt")
    test_path = os.path.join(filelists_dir, f"{args.dataset}_audio_sid_text_test.txt")

    write_filelist(train_items, train_path)
    write_filelist(val_items, val_path)
    write_filelist(test_items, test_path)

    print(f"  Train filelist: {train_path}")
    print(f"  Val filelist:   {val_path}")
    print(f"  Test filelist:  {test_path}")

    # MFA alignment (optional)
    if args.run_mfa:
        print(f"\n[Optional] Running Montreal Forced Aligner...")
        print("Note: MFA requires prepared audio and text files")
        print("Skipping MFA - run manually if needed:")
        print(f"  mfa align {wav_out_dir} english_us_arpa english_us_arpa {args.out_dir}/alignments")

    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. (Optional) Run MFA alignment for forced alignment")
    print(f"2. Extract prosody features: python extract_prosody_features.py")
    print(f"3. Start training: python train_full.py --config configs/{args.dataset}_base.json")


if __name__ == "__main__":
    main()

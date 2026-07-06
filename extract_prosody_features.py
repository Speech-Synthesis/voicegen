import os
import argparse
import json
import re
import numpy as np
try:
    import pyworld as pw
    HAS_PYWORLD = True
except ImportError:
    HAS_PYWORLD = False
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
from joblib import Parallel, delayed
from utils import filter_phones

HOP = 256
SR = 22050
FRAME_MS = 1000 * HOP / SR  # ~11.6 ms

def parse_textgrid_intervals(tg_path):
    """
    Robust custom TextGrid parser that doesn't rely on external libraries.
    Extracts the interval triples (phone, xmin, xmax) from the 'phones' or 'words' tier.
    """
    if not os.path.exists(tg_path):
        return []
    
    with open(tg_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Find the phone tier item
    # We look for "class = \"IntervalTier\"" and name = "phones"
    items = re.split(r'item\s*\[\d+\]\s*:', content)
    phone_item = None
    for item in items:
        if ('class = "IntervalTier"' in item or 'class = "IntervalTier"' in item.replace(" ", "")) and \
           ('name = "phones"' in item or 'name = "phones"' in item.replace(" ", "")):
            phone_item = item
            break
    
    if phone_item is None:
        # Fallback to the first tier if "phones" tier isn't found
        for item in items:
            if 'class = "IntervalTier"' in item or 'class = "IntervalTier"' in item.replace(" ", ""):
                phone_item = item
                break
                
    if phone_item is None:
        return []

    # Extract all intervals from the chosen tier
    intervals = []
    # Find block of intervals [num]:
    interval_blocks = re.split(r'intervals\s*\[\d+\]\s*:', phone_item)
    for block in interval_blocks[1:]: # Skip the metadata block before the first interval
        xmin_match = re.search(r'xmin\s*=\s*([\d\.]+)', block)
        xmax_match = re.search(r'xmax\s*=\s*([\d\.]+)', block)
        text_match = re.search(r'text\s*=\s*"([^"]*)"', block)
        if xmin_match and xmax_match and text_match:
            xmin = float(xmin_match.group(1))
            xmax = float(xmax_match.group(1))
            text = text_match.group(1)
            intervals.append((text, xmin, xmax))
            
    return intervals

def frame_features(wav_path):
    """Extract frame-level F0 using PyWorld and Energy using Librosa."""
    if not HAS_LIBROSA or not HAS_PYWORLD:
        raise RuntimeError("pyworld and librosa are required for frame_features. "
                           "Install them with: pip install pyworld librosa")
    wav, _ = librosa.load(wav_path, sr=SR)
    x = wav.astype(np.float64)
    # pyworld DIO + StoneMask
    f0, t = pw.dio(x, SR, frame_period=FRAME_MS)
    f0 = pw.stonemask(x, f0, t, SR)  # [n_frames]
    energy = librosa.feature.rms(y=wav, frame_length=1024, hop_length=HOP)[0]  # [n_frames']
    n = min(len(f0), len(energy))
    return f0[:n], energy[:n]

def phoneme_reduce(f0, energy, intervals):
    """
    Reduce frame features to per-phoneme values matching interval durations.
    Filters phonemes consistent with VITS text processing rules.
    """
    rows = []
    voiced_mask = []
    filtered_phones_list = []
    
    for phone, s, e in intervals:
        cleaned_phone = phone.lower().strip()
        # Filter silent phonemes exactly like VITS
        if cleaned_phone in ["sil", "sp", "spn", ""]:
            continue
            
        a = int(s * SR / HOP)
        b = max(int(s * SR / HOP) + 1, int(e * SR / HOP))
        
        f0_seg = f0[a:b]
        en_seg = energy[a:b]
        
        # Check if voiced frames exist in this interval
        v = f0_seg > 0
        if v.any():
            pitch = np.log(f0_seg[v]).mean()
            has_voiced = 1
        else:
            pitch = 0.0 # Placeholder, will be normalized later
            has_voiced = 0
            
        en_val = np.log(en_seg.mean() + 1e-8)
        dur_val = np.log1p(e - s)
        
        rows.append([pitch, en_val, dur_val])
        voiced_mask.append(has_voiced)
        filtered_phones_list.append(cleaned_phone)
        
    return np.asarray(rows, np.float32), np.asarray(voiced_mask, np.uint8), " ".join(filtered_phones_list)

def process_single(wav_path, tg_path, speaker_id, out_dir):
    """Process a single utterance, return raw statistics and info."""
    basename = os.path.basename(wav_path).replace(".wav", "")
    try:
        intervals = parse_textgrid_intervals(tg_path)
        if not intervals:
            return None
            
        f0, energy = frame_features(wav_path)
        features, voiced_mask, filtered_phoneme_string = phoneme_reduce(f0, energy, intervals)
        
        if len(features) == 0:
            return None
            
        # Temporarily save raw features
        raw_npy_path = os.path.join(out_dir, f"{basename}_raw.npy")
        voiced_npy_path = os.path.join(out_dir, f"{basename}.voiced.npy")
        np.save(raw_npy_path, features)
        np.save(voiced_npy_path, voiced_mask)
        
        return {
            "basename": basename,
            "speaker_id": speaker_id,
            "phonemes": filtered_phoneme_string,
            "raw_path": raw_npy_path,
            "voiced_path": voiced_npy_path,
            "features": features, # keep for fast stat calculation
            "voiced_mask": voiced_mask
        }
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def validate(filelist):
    """Assert phoneme count matches prosody sequence length."""
    bad = []
    with open(filelist, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4:
                continue
            wav, spk, phones, npy = parts[0], parts[1], parts[2], parts[3]
            try:
                data = np.load(npy)
                if len(phones.split()) != data.shape[0]:
                    bad.append((wav, len(phones.split()), data.shape[0]))
            except Exception as e:
                bad.append((wav, f"Error: {e}", 0))
    
    if bad:
        print(f"WARNING: {len(bad)} validation failures detected!")
        for b in bad[:5]:
            print(f"Filelist entry mismatch: {b[0]} | text len: {b[1]} | npy len: {b[2]}")
        return False
    else:
        print("All validation checks passed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Prosody extraction pipeline.")
    parser.add_argument("--in_dir", type=str, required=True, help="Path to base data dir, e.g. data/processed/vctk")
    parser.add_argument("--out_dir", type=str, required=True, help="Output prosody directory")
    parser.add_argument("--filelists", type=str, required=True, help="Output filelists directory")
    parser.add_argument("--sr", type=int, default=22050, help="Sampling rate")
    parser.add_argument("--hop", type=int, default=256, help="Hop size")
    parser.add_argument("--f0_method", type=str, default="pyworld", help="F0 extraction method")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of worker jobs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.filelists, exist_ok=True)

    # Search for wavs and alignments
    wavs_dir = os.path.join(args.in_dir, "wavs")
    tg_dir = os.path.join(args.in_dir, "alignments")
    
    if not os.path.exists(wavs_dir):
        print(f"Wavs directory {wavs_dir} not found. Creating mock directories for verification...")
        os.makedirs(wavs_dir, exist_ok=True)
        os.makedirs(tg_dir, exist_ok=True)

    # Let's list files
    wav_files = []
    for root, _, files in os.walk(wavs_dir):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))
                
    print(f"Found {len(wav_files)} wav files to process.")
    
    if len(wav_files) == 0:
        # Create a mock pair for verification purposes if directory is empty
        mock_wav = os.path.join(wavs_dir, "mock_01.wav")
        mock_tg = os.path.join(tg_dir, "mock_01.TextGrid")
        
        # Write mock 1s wav
        sr = args.sr
        t = np.linspace(0, 1, sr, endpoint=False)
        # 220Hz sine wave representing F0
        y = 0.5 * np.sin(2 * np.pi * 220 * t)
        import scipy.io.wavfile as wavfile
        wavfile.write(mock_wav, sr, (y * 32767).astype(np.int16))
        
        # Write mock TextGrid
        with open(mock_tg, "w", encoding="utf-8") as f:
            f.write("""File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 1
item [1]:
    class = "IntervalTier"
    name = "phones"
    xmin = 0
    xmax = 1.0
    intervals: size = 3
    intervals [1]:
        xmin = 0
        xmax = 0.2
        text = "sil"
    intervals [2]:
        xmin = 0.2
        xmax = 0.8
        text = "ah"
    intervals [3]:
        xmin = 0.8
        xmax = 1.0
        text = "sil"
""")
        wav_files = [mock_wav]
        print(f"Generated mock wav {mock_wav} and TextGrid {mock_tg} for smoke test.")

    # Process all files in parallel
    tasks = []
    for wav_path in wav_files:
        basename = os.path.basename(wav_path).replace(".wav", "")
        # MFA might place TextGrid in nested speaker dirs or flat dir
        # Let's search flat first, then matching nested path
        tg_name = f"{basename}.TextGrid"
        tg_path = os.path.join(tg_dir, tg_name)
        if not os.path.exists(tg_path):
            # Check nested speaker subdirectories
            rel_parent = os.path.basename(os.path.dirname(wav_path))
            tg_path = os.path.join(tg_dir, rel_parent, tg_name)
            
        # extract speaker_id from path
        # Assume VCTK speaker naming convention: e.g. p225, or parent folder name
        speaker_id = os.path.basename(os.path.dirname(wav_path))
        if not speaker_id or speaker_id == "wavs":
            speaker_id = "0"
            
        tasks.append((wav_path, tg_path, speaker_id))
        
    print("Extracting raw prosody features...")
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_single)(w, tg, spk, args.out_dir) for w, tg, spk in tasks
    )
    results = [r for r in results if r is not None]
    print(f"Successfully extracted features for {len(results)} utterances.")
    
    if not results:
        print("No valid features extracted. Exiting.")
        return

    # Pass 1: Compute Speaker stats & Global Duration stats
    speaker_stats = {}
    global_durations = []
    
    for r in results:
        spk = r["speaker_id"]
        feats = r["features"]
        voiced = r["voiced_mask"]
        
        if spk not in speaker_stats:
            speaker_stats[spk] = {"pitches": [], "energies": []}
            
        # Voiced-only pitches
        voiced_pitches = feats[voiced == 1, 0]
        speaker_stats[spk]["pitches"].extend(voiced_pitches.tolist())
        speaker_stats[spk]["energies"].extend(feats[:, 1].tolist())
        global_durations.extend(feats[:, 2].tolist())
        
    # Compute speaker stats
    stats_json = {}
    for spk, data in speaker_stats.items():
        pitches = np.array(data["pitches"])
        energies = np.array(data["energies"])
        
        stats_json[spk] = {
            "pitch_mean": float(pitches.mean()) if len(pitches) > 0 else 0.0,
            "pitch_std": float(pitches.std()) if len(pitches) > 1 else 1.0,
            "energy_mean": float(energies.mean()) if len(energies) > 0 else 0.0,
            "energy_std": float(energies.std()) if len(energies) > 1 else 1.0,
        }
        
    global_durations = np.array(global_durations)
    dur_mean = float(global_durations.mean()) if len(global_durations) > 0 else 0.0
    dur_std = float(global_durations.std()) if len(global_durations) > 1 else 1.0
    
    stats_json["global"] = {
        "duration_mean": dur_mean,
        "duration_std": dur_std
    }
    
    # Save statistics
    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2)
    print(f"Saved normalization statistics to {stats_path}")

    # Pass 2: Normalize and write final files
    filelist_entries = []
    for r in results:
        basename = r["basename"]
        spk = r["speaker_id"]
        feat_path = r["raw_path"]
        voiced_mask = r["voiced_mask"]
        phonemes = r["phonemes"]
        
        # Load raw features
        feats = np.load(feat_path)
        
        # Normalize
        spk_stats = stats_json[spk]
        
        # Pitch
        voiced = voiced_mask == 1
        if voiced.any():
            feats[voiced, 0] = (feats[voiced, 0] - spk_stats["pitch_mean"]) / spk_stats["pitch_std"]
        # Set unvoiced phonemes to 0
        feats[~voiced, 0] = 0.0
        
        # Energy
        feats[:, 1] = (feats[:, 1] - spk_stats["energy_mean"]) / spk_stats["energy_std"]
        
        # Duration (global)
        feats[:, 2] = (feats[:, 2] - dur_mean) / dur_std
        
        # Overwrite with normalized features
        final_npy_path = os.path.join(args.out_dir, f"{basename}.npy")
        np.save(final_npy_path, feats)
        
        # Remove temporary raw npy file
        if os.path.exists(feat_path):
            os.remove(feat_path)
            
        # Filelist format: wav_path|speaker_id|phoneme_string|prosody_npy_path
        # We need a relative path or direct absolute path for the wav
        # Use absolute path for robustness
        wav_abs_path = os.path.abspath(os.path.join(wavs_dir, f"{basename}.wav"))
        filelist_entries.append(f"{wav_abs_path}|{spk}|{phonemes}|{os.path.abspath(final_npy_path)}")

    # Write filelists
    train_filelist = os.path.join(args.filelists, "vctk_audio_sid_text_train.txt.prosody")
    val_filelist = os.path.join(args.filelists, "vctk_audio_sid_text_val.txt.prosody")
    
    # Split train/val (95% train, 5% val)
    np.random.seed(1234)
    np.random.shuffle(filelist_entries)
    split_idx = int(0.95 * len(filelist_entries))
    
    train_data = filelist_entries[:split_idx]
    val_data = filelist_entries[split_idx:]
    
    # Handle tiny/mock dataset edge case
    if len(val_data) == 0:
        val_data = train_data
        
    with open(train_filelist, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data) + "\n")
        
    with open(val_filelist, "w", encoding="utf-8") as f:
        f.write("\n".join(val_data) + "\n")
        
    print(f"Wrote filelists: {train_filelist} and {val_filelist}")
    
    # Validate the generated filelists
    print("Validating train filelist...")
    validate(train_filelist)
    print("Validating validation filelist...")
    validate(val_filelist)

if __name__ == "__main__":
    main()

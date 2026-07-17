# Dataset Support Analysis

**Analysis Date:** 2026-07-17
**Datasets:** LJSpeech-1.1, VCTK-Corpus-0.92, LibriTTS

---

## Executive Summary

### Current Status: ⚠️ **INCOMPLETE**

| Component | LJSpeech | VCTK | LibriTTS | Status |
|-----------|----------|------|----------|--------|
| **Preprocessing** | ✅ Full | ✅ Full | ✅ Full | Complete |
| **Training Configs** | ❌ Missing | ✅ Full | ❌ Missing | **Incomplete** |
| **Training Scripts** | ⚠️ Wrong config | ✅ Works | ❌ None | **Needs fix** |
| **Data Loaders** | ✅ Works | ✅ Works | ✅ Works | Complete |
| **Model Architecture** | ✅ Works | ✅ Works | ✅ Works | Complete |
| **Inference** | ✅ Works | ✅ Works | ✅ Works | Complete |
| **Evaluation** | ✅ Works | ✅ Works | ✅ Works | Complete |

---

## Detailed Analysis

### 1. Preprocessing (preprocess.py) ✅

**Status:** COMPLETE for all 3 datasets

**Evidence:**
```python
# Lines 230-317 in preprocess.py
class LJSpeechParser(DatasetParser):     # ✅ Implemented
class VCTKParser(DatasetParser):         # ✅ Implemented (just fixed for FLAC)
class LibriTTSParser(DatasetParser):     # ✅ Implemented

# Line 321-332
parsers = {
    'ljspeech': LJSpeechParser,
    'vctk': VCTKParser,
    'libritts': LibriTTSParser,
}
```

**Works correctly:**
```bash
✅ python preprocess.py --dataset ljspeech --data_dir data/LJSpeech-1.1 --out_dir data/processed/ljspeech
✅ python preprocess.py --dataset vctk --data_dir data/VCTK-Corpus-0.92 --out_dir data/processed/vctk
✅ python preprocess.py --dataset libritts --data_dir data/LibriTTS --out_dir data/processed/libritts
```

---

### 2. Training Configurations ❌

**Status:** INCOMPLETE - Only VCTK configs exist

**Current configs directory:**
```
configs/
├── vctk_base.json               ✅ VCTK baseline
├── vctk_full.json               ✅ VCTK research
├── vctk_prosody_pretrain.json   ✅ VCTK prosody
├── vctk_abl_nodis.json          ✅ VCTK ablation
├── vctk_abl_concat.json         ✅ VCTK ablation
└── vctk_abl_sweep_*.json        ✅ VCTK ablations

❌ ljspeech_base.json            MISSING
❌ libritts_base.json            MISSING
```

**Critical Issue:**
The configs specify dataset-specific paths:
```json
{
  "data": {
    "training_files": "filelists/vctk_audio_sid_text_train.txt",  // ← Dataset-specific
    "validation_files": "filelists/vctk_audio_sid_text_val.txt",
    "n_speakers": 109  // ← VCTK-specific (109 speakers)
  }
}
```

LJSpeech needs:
- `"training_files": "filelists/ljspeech_audio_sid_text_train.txt"`
- `"n_speakers": 1` (single speaker)

LibriTTS needs:
- `"training_files": "filelists/libritts_audio_sid_text_train.txt"`
- `"n_speakers": varies` (depends on subset used)

---

### 3. Training Job Scripts ⚠️

**Status:** INCOMPLETE - Wrong config referenced

**trainjob.sh (Line 20):**
```bash
# ❌ WRONG: Uses VCTK config for LJSpeech training
python train_full.py --config configs/vctk_base.json --model_dir checkpoints/baseline_ljspeech
#                                    ^^^^^^^^^^^^^^
#                                    Should be ljspeech_base.json
```

**Should be:**
```bash
python train_full.py --config configs/ljspeech_base.json --model_dir checkpoints/baseline_ljspeech
```

**Missing:**
- No `trainjob_libritts.sh` exists

---

### 4. Data Loaders (data_utils.py, data_utils_research.py) ✅

**Status:** COMPLETE - Dataset-agnostic

**Evidence:**
```python
# data_utils.py - Lines 7-36
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths_sid_text, hparams):
        # ✅ Loads from any filelist format: wav_path|speaker_id|text
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
```

**Works with all datasets:**
- Reads generic format: `wav_path|speaker_id|phonemes`
- No dataset-specific logic
- Speaker IDs can be 0 (LJSpeech), 0-108 (VCTK), or any (LibriTTS)

---

### 5. Model Architecture (models.py, models_research.py) ✅

**Status:** COMPLETE - Dataset-agnostic

**Evidence:**
```python
# models.py - Lines 1022-1100
class SynthesizerTrn(nn.Module):
    def __init__(self, n_vocab, ..., n_speakers, gin_channels):
        # ✅ Configurable n_speakers (can be 0, 1, or 100+)
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
```

**Works with:**
- LJSpeech: `n_speakers=1` (single speaker)
- VCTK: `n_speakers=109` (multi-speaker)
- LibriTTS: `n_speakers=X` (depends on subset)

---

### 6. Inference (inference.py) ✅

**Status:** COMPLETE - Model-agnostic

**Evidence:**
```python
# inference.py - Lines 45-161
class InferencePipeline:
    def __init__(self, config_path, checkpoint_path, device="cuda"):
        # ✅ Loads any trained model regardless of training dataset
        self.model = self._load_model(checkpoint_path)
```

**Works with models trained on any dataset:**
- Takes any checkpoint
- Reads config to determine architecture
- Dataset used for training doesn't matter

---

### 7. Evaluation (eval.py) ✅

**Status:** COMPLETE - Model-agnostic

**Evidence:**
```python
# eval.py - Lines 33-100
def compute_secs(wav_gen, wav_ref, ecapa_model):
    # ✅ Works with any audio
def compute_wer(text_gen, text_ref):
    # ✅ Works with any text
```

**Works with:**
- Any generated audio
- Any reference audio
- Dataset-independent metrics

---

## Critical Gaps

### ❌ Gap 1: Missing LJSpeech Config

**Impact:** Cannot train baseline model on LJSpeech (Stage 4)

**Required:** `configs/ljspeech_base.json`

**Differences from VCTK:**
```diff
- "training_files": "filelists/vctk_audio_sid_text_train.txt",
+ "training_files": "filelists/ljspeech_audio_sid_text_train.txt",

- "validation_files": "filelists/vctk_audio_sid_text_val.txt",
+ "validation_files": "filelists/ljspeech_audio_sid_text_val.txt",

- "n_speakers": 109,
+ "n_speakers": 1,
```

### ❌ Gap 2: Missing LibriTTS Config

**Impact:** Cannot train on LibriTTS for robustness

**Required:** `configs/libritts_base.json`

**Differences from VCTK:**
```diff
- "training_files": "filelists/vctk_audio_sid_text_train.txt",
+ "training_files": "filelists/libritts_audio_sid_text_train.txt",

- "validation_files": "filelists/vctk_audio_sid_text_val.txt",
+ "validation_files": "filelists/libritts_audio_sid_text_val.txt",

- "n_speakers": 109,
+ "n_speakers": 100,  // train-clean-100 has ~100 speakers
```

### ⚠️ Gap 3: Wrong Config in trainjob.sh

**Impact:** LJSpeech training job uses wrong config

**Required:** Fix line 20 in `trainjob.sh`

### ❌ Gap 4: Missing LibriTTS Training Script

**Impact:** No PBS script for LibriTTS training

**Required:** `trainjob_libritts.sh`

---

## Summary

### What Works ✅

1. **Preprocessing:** All 3 datasets can be preprocessed
2. **Data Loading:** Filelists from all 3 datasets can be loaded
3. **Model:** Architecture supports single-speaker and multi-speaker
4. **Inference:** Works with any trained model
5. **Evaluation:** Works with any model output

### What's Missing ❌

1. **Config for LJSpeech:** No `ljspeech_base.json`
2. **Config for LibriTTS:** No `libritts_base.json`
3. **Wrong config reference:** `trainjob.sh` uses VCTK config
4. **Missing script:** No `trainjob_libritts.sh`

---

## Impact Assessment

### Can you train on LJSpeech right now? ⚠️ PARTIALLY

**Currently (BROKEN):**
```bash
qsub trainjob.sh
# ❌ Will fail: Uses vctk_base.json which expects VCTK filelists
# ❌ Will look for filelists/vctk_audio_sid_text_train.txt
# ❌ But LJSpeech preprocessing creates filelists/ljspeech_audio_sid_text_train.txt
```

**Workaround (MANUAL):**
```bash
# Must manually specify correct config (after creating it)
python train_full.py --config configs/ljspeech_base.json --model_dir checkpoints/ljspeech
```

### Can you train on VCTK right now? ✅ YES

```bash
qsub trainjob_vctk.sh
# ✅ Works perfectly - all configs exist
```

### Can you train on LibriTTS right now? ❌ NO

```bash
# No config, no script
```

---

## Recommendation

### Priority 1: Create Missing Configs

Create these files:
1. `configs/ljspeech_base.json` (for Stage 4)
2. `configs/libritts_base.json` (for robustness training)

### Priority 2: Fix trainjob.sh

Update line 20:
```bash
python train_full.py --config configs/ljspeech_base.json --model_dir checkpoints/baseline_ljspeech
```

### Priority 3: Create trainjob_libritts.sh

For LibriTTS training on HPC.

---

## Conclusion

**Current Answer to "Will our whole code work for these three datasets?"**

| Dataset | Can Preprocess? | Can Train? | Can Infer? | Overall |
|---------|----------------|------------|------------|---------|
| **LJSpeech-1.1** | ✅ Yes | ⚠️ Needs config | ✅ Yes | ⚠️ **Incomplete** |
| **VCTK-0.92** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ **Complete** |
| **LibriTTS** | ✅ Yes | ❌ Needs config | ✅ Yes | ❌ **Missing** |

**Overall Status:** ⚠️ **VCTK-only currently complete**

To fully support all three datasets, we need to create the missing configuration files and fix the training script reference.

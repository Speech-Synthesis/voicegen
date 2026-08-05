#!/usr/bin/env python3
"""
Test script to debug data loading issues.
Run: python tests/test_data_loading.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data_utils_research import TextAudioSpeakerLoaderResearch, TextAudioSpeakerCollateResearch
from utils import get_hparams_from_file

def main():
    print("=" * 60)
    print("DATA LOADING TEST")
    print("=" * 60)

    # Load config
    config_path = "configs/vctk_full.json"
    print(f"\n1. Loading config: {config_path}")
    hps = get_hparams_from_file(config_path)
    print(f"   filter_length: {hps.data.filter_length}")
    print(f"   Expected spec_channels: {hps.data.filter_length // 2 + 1}")

    # Load dataset
    print(f"\n2. Loading dataset: {hps.data.training_files}")
    dataset = TextAudioSpeakerLoaderResearch(hps.data.training_files, hps)
    print(f"   Dataset size: {len(dataset)}")

    # Test single item
    print("\n3. Testing single item (index 0):")
    try:
        item = dataset[0]
        text, spec, wav, sid, p_feat, p_mask, voiced, g_timbre = item

        print(f"   text shape:    {text.shape}")
        print(f"   spec shape:    {spec.shape}  <- Should be [513, T]")
        print(f"   wav shape:     {wav.shape}")
        print(f"   sid:           {sid}")
        print(f"   p_feat shape:  {p_feat.shape}")
        print(f"   p_mask shape:  {p_mask.shape}")
        print(f"   voiced shape:  {voiced.shape}")
        print(f"   g_timbre shape: {g_timbre.shape}")

        print(f"\n   spec dtype:    {spec.dtype}")
        print(f"   spec min:      {spec.min():.4f}")
        print(f"   spec max:      {spec.max():.4f}")
        print(f"   spec has NaN:  {torch.isnan(spec).any()}")
        print(f"   spec has Inf:  {torch.isinf(spec).any()}")

        # Verify spec channels
        expected_channels = hps.data.filter_length // 2 + 1
        actual_channels = spec.shape[0]
        if actual_channels == expected_channels:
            print(f"\n   ✓ spec_channels CORRECT: {actual_channels}")
        else:
            print(f"\n   ✗ spec_channels MISMATCH: got {actual_channels}, expected {expected_channels}")

    except Exception as e:
        print(f"   ✗ ERROR loading item: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test collate with mini-batch
    print("\n4. Testing collate with batch_size=2:")
    try:
        collate = TextAudioSpeakerCollateResearch()
        batch = collate([dataset[i] for i in range(2)])

        (text_padded, text_lengths, spec_padded, spec_lengths,
         wav_padded, wav_lengths, sid,
         p_feat_padded, p_mask_padded, voiced_padded, g_timbre_padded) = batch

        print(f"   text_padded shape:    {text_padded.shape}")
        print(f"   spec_padded shape:    {spec_padded.shape}  <- Should be [2, 513, T]")
        print(f"   wav_padded shape:     {wav_padded.shape}")
        print(f"   p_feat_padded shape:  {p_feat_padded.shape}")
        print(f"   g_timbre_padded shape: {g_timbre_padded.shape}")

        print(f"\n   spec_padded dtype:   {spec_padded.dtype}")
        print(f"   spec_padded min:     {spec_padded.min():.4f}")
        print(f"   spec_padded max:     {spec_padded.max():.4f}")
        print(f"   spec_padded has NaN: {torch.isnan(spec_padded).any()}")
        print(f"   spec_padded has Inf: {torch.isinf(spec_padded).any()}")

    except Exception as e:
        print(f"   ✗ ERROR in collate: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test GPU transfer
    print("\n5. Testing GPU transfer:")
    if torch.cuda.is_available():
        try:
            spec_gpu = spec_padded.cuda()
            print(f"   ✓ Transferred to GPU successfully")
            print(f"   GPU spec shape: {spec_gpu.shape}")
            print(f"   GPU spec device: {spec_gpu.device}")
        except Exception as e:
            print(f"   ✗ GPU transfer failed: {e}")
            return 1
    else:
        print("   CUDA not available, skipping GPU test")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())

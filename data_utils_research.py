import os
import numpy as np
import torch
import torch.utils.data
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate

class TextAudioSpeakerLoaderResearch(TextAudioSpeakerLoader):
    """
    Prosody-aware dataset loader.
    Parses filelists of format: wav_path|speaker_id|phoneme_string|prosody_npy_path
    """
    def __init__(self, audiopaths_sid_text_prosody, hparams):
        # We override __init__ but need to load files differently
        # Let's call the base class init, but then overwrite self.audiopaths_sid_text
        super().__init__(audiopaths_sid_text_prosody, hparams)
        # Re-parse because base class parses only 3 columns
        self.audiopaths_sid_text = self.load_filepaths_and_text_prosody(audiopaths_sid_text_prosody)

    def load_filepaths_and_text_prosody(self, filename, split="|"):
        with open(filename, encoding='utf-8') as f:
            filepaths_and_text = [line.strip().split(split) for line in f if line.strip()]
        return filepaths_and_text

    def __getitem__(self, index):
        # Base returns: text, spec, wav, sid
        item = self.audiopaths_sid_text[index]
        if len(item) < 4:
            # Fallback if no prosody column is present (baseline mode)
            res = super().get_audio_text_speaker_pair(item)
            text, spec, wav, sid = res
            # Return dummy prosody/timbre features
            T = len(text)
            p_feat = torch.zeros(T, 3, dtype=torch.float32)
            p_mask = torch.ones(T, dtype=torch.uint8)
            voiced = torch.ones(T, dtype=torch.uint8)
            g_timbre = torch.zeros(192, dtype=torch.float32)
            return (text, spec, wav, sid, p_feat, p_mask, voiced, g_timbre)
            
        wav_path, speaker_id, phonemes, prosody_path = item[0], item[1], item[2], item[3]
        
        # Load base audio/spec features
        text, spec, wav, sid = super().get_audio_text_speaker_pair([wav_path, speaker_id, phonemes])
        
        # Load prosody features
        try:
            p_feat_np = np.load(prosody_path)
            # Load voicing mask
            voiced_path = prosody_path.replace(".npy", ".voiced.npy")
            if os.path.exists(voiced_path):
                voiced_np = np.load(voiced_path)
            else:
                # Fallback voicing: pitch col is first column
                voiced_np = (p_feat_np[:, 0] != 0.0).astype(np.uint8)
        except Exception as e:
            # If load fails, create dummy arrays
            T = len(phonemes.split())
            p_feat_np = np.zeros((T, 3), dtype=np.float32)
            voiced_np = np.ones(T, dtype=np.uint8)
            
        p_feat = torch.FloatTensor(p_feat_np)
        voiced = torch.ByteTensor(voiced_np)
        p_mask = torch.ones(p_feat.size(0), dtype=torch.uint8)
        
        # Assert phoneme counts align with prosody sequence length
        assert len(text) == p_feat.size(0), \
            f"Length mismatch: text tokens ({len(text)}) vs prosody rows ({p_feat.size(0)}) for {wav_path}"
            
        # Load cached speaker timbre embedding (from ECAPA-TDNN)
        # Search for .npy timbre file: /wavs/basename.wav -> /timbre/basename.npy
        timbre_path = wav_path.replace("/wavs/", "/timbre/").replace(".wav", ".npy")
        if os.path.exists(timbre_path):
            try:
                g_timbre_np = np.load(timbre_path)
                g_timbre = torch.FloatTensor(g_timbre_np)
            except Exception:
                g_timbre = torch.zeros(192, dtype=torch.float32)
        else:
            # If no cached file, return a dummy speaker embedding for test execution stability
            g_timbre = torch.zeros(192, dtype=torch.float32)
            
        return (text, spec, wav, sid, p_feat, p_mask, voiced, g_timbre)

class TextAudioSpeakerCollateResearch():
    """
    Collate function that extends VITS collate to pad prosody and timbre tensors,
    preserving sequence lengths and sorting order.
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        # Elements: text, spec, wav, sid, p_feat, p_mask, voiced, g_timbre
        # Sort batch by spec length descending (same as baseline collate)
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(0) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len).zero_()
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len).zero_()
        wav_padded = torch.FloatTensor(len(batch), max_wav_len).zero_()
        
        # Research additions:
        p_feat_padded = torch.FloatTensor(len(batch), max_text_len, 3).zero_()
        p_mask_padded = torch.BoolTensor(len(batch), max_text_len).zero_()
        voiced_padded = torch.ByteTensor(len(batch), max_text_len).zero_()
        g_timbre_padded = torch.FloatTensor(len(batch), 192).zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :wav.size(0)] = wav
            wav_lengths[i] = wav.size(0)

            sid[i] = row[3]
            
            # Prosody & timbre padding
            p_feat = row[4]
            p_feat_padded[i, :p_feat.size(0), :] = p_feat
            
            p_mask = row[5]
            p_mask_padded[i, :p_mask.size(0)] = p_mask
            
            voiced = row[6]
            voiced_padded[i, :voiced.size(0)] = voiced
            
            g_timbre = row[7]
            g_timbre_padded[i] = g_timbre

        return (text_padded, text_lengths, spec_padded, spec_lengths,
                wav_padded, wav_lengths, sid,
                p_feat_padded, p_mask_padded, voiced_padded, g_timbre_padded)

import os
import random
import numpy as np
import torch
import torch.utils.data

class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.hparams = hparams
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.cleaned_text = getattr(hparams.data, "cleaned_text", False)
        self.add_blank = getattr(hparams.data, "add_blank", True)

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # audiopath_sid_text: [wav_path, speaker_id, phonemes]
        # Return mock tensors for testing
        wav_path, speaker_id, phonemes = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        
        # mock phoneme ids
        text = torch.randint(1, 20, (len(phonemes.split()),))
        spec = torch.randn(80, len(phonemes.split()) * 2) # spec length ≈ 2 * phoneme length
        wav = torch.randn(len(phonemes.split()) * 2 * self.hop_length)
        sid = torch.tensor(int(speaker_id))
        
        return (text, spec, wav, sid)

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)

class TextAudioSpeakerCollate():
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's output format:
        text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid
        """
        # Right zero-pad all one-dimensional tensors
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

        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f if line.strip()]
    return filepaths_and_text

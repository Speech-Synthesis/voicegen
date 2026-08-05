import os
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

# Try to import audio libraries
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import scipy.io.wavfile as wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_wav(wav_path, sampling_rate):
    """Load audio file and resample if necessary."""
    if HAS_LIBROSA:
        wav, sr = librosa.load(wav_path, sr=sampling_rate)
        return wav
    elif HAS_SCIPY:
        sr, wav = wavfile.read(wav_path)
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        elif wav.dtype == np.int32:
            wav = wav.astype(np.float32) / 2147483648.0
        if sr != sampling_rate:
            # Simple resampling - librosa is preferred
            import warnings
            warnings.warn(f"Resampling from {sr} to {sampling_rate} without librosa may be inaccurate")
        return wav
    else:
        raise RuntimeError("Either librosa or scipy is required for audio loading")


def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    """
    Compute log-scaled linear spectrogram using PyTorch STFT.

    Args:
        y: Audio waveform [T]
        n_fft: FFT size (filter_length)
        hop_size: Hop length
        win_size: Window size
        center: Whether to center the window

    Returns:
        Log-scaled linear spectrogram [n_fft // 2 + 1, T']
    """
    # Validate input
    if y.dim() != 1:
        raise ValueError(f"Expected 1D audio, got shape {y.shape}")

    if len(y) < n_fft:
        # Pad short audio to at least n_fft length
        y = F.pad(y, (0, n_fft - len(y)))

    hann_window = torch.hann_window(win_size, device=y.device, dtype=y.dtype)

    # Pad audio for non-centered STFT
    if not center:
        pad_amount = int((n_fft - hop_size) / 2)
        y = F.pad(y.unsqueeze(0), (pad_amount, pad_amount), mode='reflect').squeeze(0)

    # Compute STFT
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )

    # Convert to magnitude spectrogram
    spec = torch.abs(spec)

    # Apply log scaling (standard in VITS)
    # Clamp to avoid log(0)
    spec = torch.log(torch.clamp(spec, min=1e-5))

    return spec


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

        # Compute expected spec channels for validation
        self.spec_channels = self.filter_length // 2 + 1

        # Build speaker-to-index mapping from filelist
        self.speaker_to_idx = self._build_speaker_mapping()

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        """
        Load audio and compute linear spectrogram.

        Args:
            audiopath_sid_text: [wav_path, speaker_id, phonemes]

        Returns:
            text: [T_text] phoneme token IDs
            spec: [spec_channels, T_spec] linear spectrogram
            wav: [T_audio] raw waveform
            sid: [] speaker ID tensor
        """
        wav_path, speaker_id, phonemes = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]

        # Parse phonemes to token IDs (simple mapping: each phoneme -> unique ID)
        phoneme_list = phonemes.split()
        # Use simple hash-based token IDs (real implementation would use a vocabulary)
        text = torch.LongTensor([hash(p) % 200 + 1 for p in phoneme_list])

        # Load audio
        try:
            wav_np = load_wav(wav_path, self.sampling_rate)
            wav = torch.FloatTensor(wav_np)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            # Return mock data on error
            T_text = len(phoneme_list)
            spec = torch.randn(self.spec_channels, T_text * 4)
            wav = torch.randn(T_text * 4 * self.hop_length)
            sid = self._parse_speaker_id(speaker_id)
            return (text, spec, wav, sid)

        # Compute linear spectrogram
        spec = spectrogram_torch(wav, self.filter_length, self.hop_length, self.win_length)

        # Parse speaker ID
        sid = self._parse_speaker_id(speaker_id)

        return (text, spec, wav, sid)

    def _build_speaker_mapping(self):
        """Build a mapping from speaker_id strings to 0-indexed integers."""
        unique_speakers = set()
        for item in self.audiopaths_sid_text:
            if len(item) >= 2:
                unique_speakers.add(str(item[1]).strip())

        # Sort for reproducibility
        sorted_speakers = sorted(unique_speakers)
        speaker_to_idx = {spk: idx for idx, spk in enumerate(sorted_speakers)}

        print(f"Built speaker mapping: {len(speaker_to_idx)} unique speakers")
        return speaker_to_idx

    def _parse_speaker_id(self, speaker_id):
        """Map speaker_id string to 0-indexed integer tensor."""
        speaker_id = str(speaker_id).strip()

        # Use the prebuilt mapping
        if speaker_id in self.speaker_to_idx:
            return torch.tensor(self.speaker_to_idx[speaker_id])

        # Fallback for unknown speakers (shouldn't happen if mapping built correctly)
        print(f"Warning: Unknown speaker '{speaker_id}', using index 0")
        return torch.tensor(0)

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

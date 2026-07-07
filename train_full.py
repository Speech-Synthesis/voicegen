"""
Full Model Training Script
Supports both baseline VITS and research model with prosody/fusion/disentanglement

Stages:
- Stage 4: Single-speaker training (LJSpeech)
- Stage 5: Multi-speaker training (VCTK)
- Stage 9R: Full research model training
"""

import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self, *args, **kwargs): pass

from utils import get_hparams_from_file, save_checkpoint, load_checkpoint
from models import SynthesizerTrn, MultiPeriodDiscriminator
from models_research import SynthesizerTrnResearch
from disentangle_loss import build_disentangle
from data_utils_research import TextAudioSpeakerLoaderResearch, TextAudioSpeakerCollateResearch

# Set random seed for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)


# ============================================================================
# VITS Loss Functions
# ============================================================================

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    KL divergence loss between posterior and prior

    Args:
        z_p: Sampled latent from prior [B, C, T]
        logs_q: Log variance from posterior encoder [B, C, T]
        m_p: Mean from text encoder (prior) [B, C, T]
        logs_p: Log variance from text encoder (prior) [B, C, T]
        z_mask: Mask [B, 1, T]

    Returns:
        KL divergence loss (scalar)
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


def feature_loss(fmap_r, fmap_g):
    """
    Feature matching loss between real and generated feature maps

    Args:
        fmap_r: List of feature maps from real audio
        fmap_g: List of feature maps from generated audio

    Returns:
        Feature matching loss (scalar)
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def generator_loss(disc_outputs):
    """
    GAN generator loss (hinge loss)

    Args:
        disc_outputs: List of discriminator outputs on generated audio

    Returns:
        Generator adversarial loss (scalar)
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    GAN discriminator loss (hinge loss)

    Args:
        disc_real_outputs: List of discriminator outputs on real audio
        disc_generated_outputs: List of discriminator outputs on generated audio

    Returns:
        Discriminator loss (scalar), real losses, fake losses
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """
    Compute mel spectrogram using PyTorch

    Args:
        y: Input waveform [B, 1, T] or [B, T]
        n_fft: FFT size
        num_mels: Number of mel bins
        sampling_rate: Sample rate
        hop_size: Hop length
        win_size: Window size
        fmin: Minimum frequency
        fmax: Maximum frequency
        center: Whether to center the window

    Returns:
        Mel spectrogram [B, num_mels, T']
    """
    if torch.min(y) < -1.0:
        print(f"Warning: min value is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"Warning: max value is {torch.max(y)}")

    # Create mel filterbank
    mel_basis = torch.from_numpy(
        librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
    ).float().to(y.device)

    hann_window = torch.hann_window(win_size).to(y.device)

    # Remove channel dimension if present
    if y.dim() == 3:
        y = y.squeeze(1)

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

    # Convert to magnitude
    spec = torch.abs(spec)

    # Apply mel filterbank
    mel = torch.matmul(mel_basis, spec)

    # Convert to log scale
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel


def librosa_mel_fn(sr, n_fft, n_mels, fmin, fmax):
    """Create mel filterbank using librosa (for compatibility)"""
    import librosa
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)


# ============================================================================
# Mock Dataset (for testing without real data)
# ============================================================================

class MockFullDataset(Dataset):
    """Mock dataset for testing"""
    def __init__(self, size=100, segment_size=8192, spec_channels=80):
        self.size = size
        self.segment_size = segment_size
        self.spec_channels = spec_channels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        T_text = np.random.randint(20, 80)
        T_spec = T_text * 2 + np.random.randint(-5, 5)
        T_spec = max(10, T_spec)

        text = torch.randint(1, 20, (T_text,), dtype=torch.long)
        spec = torch.randn(self.spec_channels, T_spec, dtype=torch.float32)
        wav = torch.randn(self.segment_size, dtype=torch.float32)
        sid = torch.tensor(np.random.randint(0, 109), dtype=torch.long)

        p_feat = torch.randn(T_text, 3, dtype=torch.float32)
        voiced = torch.ByteTensor((torch.rand(T_text) > 0.4).byte())
        p_feat[voiced == 0, 0] = 0.0
        p_mask = torch.ones(T_text, dtype=torch.uint8)

        g_timbre = torch.randn(192, dtype=torch.float32)

        return (text, spec, wav, sid, p_feat, p_mask, voiced, g_timbre)


# ============================================================================
# Main Training Function
# ============================================================================

def train():
    parser = argparse.ArgumentParser(description="Joint training (VITS + fusion + disentanglement)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_dir", type=str, default="checkpoints/full", help="Directory to save checkpoints")
    parser.add_argument("--pretrain_prosody_pth", type=str, default=None, help="Pretrained prosody encoder weights")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--force_mock", action="store_true", help="Force using mock dataset")
    args = parser.parse_args()

    # Load config
    os.makedirs(args.model_dir, exist_ok=True)
    hps = get_hparams_from_file(args.config)

    # Setup logging
    writer = SummaryWriter(log_dir=os.path.join(args.model_dir, "logs"))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========================================================================
    # Dataset & DataLoader
    # ========================================================================
    use_mock = args.force_mock or not os.path.exists(hps.data.training_files)

    if use_mock:
        print("Using MOCK dataset for training.")
        train_dataset = MockFullDataset(size=200, segment_size=hps.train.segment_size)
        val_dataset = MockFullDataset(size=20, segment_size=hps.train.segment_size)
        collate_fn = TextAudioSpeakerCollateResearch()
    else:
        print(f"Using real dataset from: {hps.data.training_files}")
        train_dataset = TextAudioSpeakerLoaderResearch(hps.data.training_files, hps)
        val_dataset = TextAudioSpeakerLoaderResearch(hps.data.validation_files, hps)
        collate_fn = TextAudioSpeakerCollateResearch()

    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hps.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )

    # ========================================================================
    # Models
    # ========================================================================

    # Check if research model
    use_research = hasattr(hps, 'research') and hps.research.use_prosody_encoder

    if use_research:
        print("Initializing Research Model (with prosody encoder + fusion + disentanglement)")
        net_g = SynthesizerTrnResearch(
            n_vocab=getattr(hps.data, 'n_vocab', 200),
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            inter_channels=hps.model.inter_channels,
            hidden_channels=hps.model.hidden_channels,
            filter_channels=hps.model.filter_channels,
            n_heads=hps.model.n_heads,
            n_layers=hps.model.n_layers,
            kernel_size=hps.model.kernel_size,
            p_dropout=hps.model.p_dropout,
            resblock=hps.model.resblock,
            resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
            resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
            upsample_rates=hps.model.upsample_rates,
            upsample_initial_channel=hps.model.upsample_initial_channel,
            upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
            n_speakers=hps.data.n_speakers,
            gin_channels=hps.model.gin_channels,
            research_cfg=hps.research
        ).to(device)
    else:
        print("Initializing Baseline VITS Model")
        net_g = SynthesizerTrn(
            n_vocab=getattr(hps.data, 'n_vocab', 200),
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            inter_channels=hps.model.inter_channels,
            hidden_channels=hps.model.hidden_channels,
            filter_channels=hps.model.filter_channels,
            n_heads=hps.model.n_heads,
            n_layers=hps.model.n_layers,
            kernel_size=hps.model.kernel_size,
            p_dropout=hps.model.p_dropout,
            resblock=hps.model.resblock,
            resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
            resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
            upsample_rates=hps.model.upsample_rates,
            upsample_initial_channel=hps.model.upsample_initial_channel,
            upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
            n_speakers=hps.data.n_speakers,
            gin_channels=hps.model.gin_channels,
            use_sdp=True
        ).to(device)

    # Discriminator
    net_d = MultiPeriodDiscriminator().to(device)

    # Load pretrained prosody encoder if specified
    if use_research and args.pretrain_prosody_pth and os.path.exists(args.pretrain_prosody_pth):
        print(f"Loading pretrained prosody encoder from: {args.pretrain_prosody_pth}")
        checkpoint = torch.load(args.pretrain_prosody_pth, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        net_g.prosody_enc.load_state_dict(state_dict, strict=True)
    elif use_research:
        print("Prosody encoder initialized with random weights")

    # ========================================================================
    # Optimizers
    # ========================================================================

    optimizer_g = torch.optim.AdamW(
        net_g.parameters(),
        lr=hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )

    optimizer_d = torch.optim.AdamW(
        net_d.parameters(),
        lr=hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )

    # Disentanglement loss
    dis_loss = None
    optimizer_mine = None
    if use_research and hasattr(hps.research, 'disentangle_loss'):
        dis_loss = build_disentangle(hps.research)
        if dis_loss is not None:
            dis_loss = dis_loss.to(device)

            # MINE specific optimizer
            if hps.research.disentangle_loss == "mine":
                print("Initializing MINE statistics network optimizer")
                optimizer_mine = torch.optim.Adam(
                    dis_loss.T.parameters(),
                    lr=1e-4,
                    betas=(0.5, 0.9)
                )

    # AMP scaler
    scaler = GradScaler(enabled=hps.train.fp16_run)

    # Resume from checkpoint
    step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        net_g, optimizer_g, _, step = load_checkpoint(args.resume, net_g, optimizer_g)
        # Try to load discriminator
        disc_path = args.resume.replace("G_", "D_")
        if os.path.exists(disc_path):
            net_d, optimizer_d, _, _ = load_checkpoint(disc_path, net_d, optimizer_d)

    # ========================================================================
    # Training Loop
    # ========================================================================

    print("Starting training loop...")
    total_steps = hps.train.total_steps if hasattr(hps.train, 'total_steps') else 100000
    if use_mock:
        total_steps = min(total_steps, 100)

    epoch = 0
    while step < total_steps:
        epoch += 1
        net_g.train()
        net_d.train()

        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch
            (x_padded, x_lengths, spec_padded, spec_lengths,
             wav_padded, wav_lengths, sid,
             p_feat_padded, p_mask_padded, voiced_padded, g_timbre_padded) = [
                b.to(device) if torch.is_tensor(b) else b for b in batch
            ]

            # ================================================================
            # Discriminator Step
            # ================================================================
            optimizer_d.zero_grad()

            with autocast(enabled=hps.train.fp16_run):
                # Generator forward (detached for discriminator training)
                if use_research:
                    outputs, extras = net_g(x_padded, x_lengths, spec_padded, spec_lengths,
                                           g_timbre_padded, p_feat_padded, p_mask_padded, sid)
                else:
                    outputs = net_g(x_padded, x_lengths, spec_padded, spec_lengths, sid=sid)

                y_hat = outputs[0]  # Generated waveform

                # Get real waveform segment (same length as generated)
                y = wav_padded.unsqueeze(1)  # [B, 1, T]

                # Discriminator on real and fake
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)

            scaler.scale(loss_disc).backward()
            scaler.unscale_(optimizer_d)
            grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), 1000.0)
            scaler.step(optimizer_d)

            # ================================================================
            # MINE Statistics Network Step (if using MINE)
            # ================================================================
            mine_mi = 0.0
            if use_research and hps.research.disentangle_loss == "mine" and dis_loss is not None and step > 0:
                net_g.eval()
                with torch.no_grad():
                    p_emb = net_g.prosody_enc(p_feat_padded, p_mask_padded)
                net_g.train()

                # Multiple discriminator updates
                for _ in range(3):
                    mine_mi = dis_loss.update_statistics_net(g_timbre_padded, p_emb, p_mask_padded, optimizer_mine)

            # ================================================================
            # Generator Step
            # ================================================================
            optimizer_g.zero_grad()

            with autocast(enabled=hps.train.fp16_run):
                # Forward pass
                if use_research:
                    outputs, extras = net_g(x_padded, x_lengths, spec_padded, spec_lengths,
                                           g_timbre_padded, p_feat_padded, p_mask_padded, sid)
                    g_t, p, p_mask_out, attn_w = extras
                else:
                    outputs = net_g(x_padded, x_lengths, spec_padded, spec_lengths, sid=sid)

                # Unpack outputs
                y_hat, l_length, z, y_mask, x_mask, (m_p, logs_p, m_q, logs_q) = outputs

                # Compute mel spectrograms
                y_mel = mel_spectrogram_torch(
                    y.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )

                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )

                # ============================================================
                # VITS Losses
                # ============================================================

                # 1. Mel reconstruction loss
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45

                # 2. KL divergence loss
                loss_kl = kl_loss(z, logs_q, m_p, logs_p, y_mask)

                # 3. Duration loss (from SDP)
                loss_duration = torch.sum(l_length.float())

                # 4. GAN losses
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                # Total VITS loss
                loss_vits = loss_gen + loss_fm + loss_mel + loss_kl + loss_duration

                # 5. Disentanglement loss (research only)
                loss_dis_total = torch.tensor(0.0, device=device)
                if use_research and dis_loss is not None:
                    loss_dis_total = dis_loss(g_t, p, p_mask_out)

                # Total generator loss
                loss_total = loss_vits + hps.research.disentangle_weight * loss_dis_total if use_research else loss_vits

            # Backward and optimize
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer_g)
            grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), 1000.0)
            scaler.step(optimizer_g)
            scaler.update()

            # ================================================================
            # Logging
            # ================================================================
            if step % hps.train.log_interval == 0:
                lr = optimizer_g.param_groups[0]['lr']
                print(f"Step {step} | Epoch {epoch} | "
                      f"Loss Total: {loss_total.item():.3f} | "
                      f"Loss Mel: {loss_mel.item():.3f} | "
                      f"Loss KL: {loss_kl.item():.3f} | "
                      f"Loss Gen: {loss_gen.item():.3f} | "
                      f"Loss Disc: {loss_disc.item():.3f}")

                # TensorBoard logging
                writer.add_scalar("loss/total", loss_total.item(), step)
                writer.add_scalar("loss/mel", loss_mel.item(), step)
                writer.add_scalar("loss/kl", loss_kl.item(), step)
                writer.add_scalar("loss/duration", loss_duration.item(), step)
                writer.add_scalar("loss/fm", loss_fm.item(), step)
                writer.add_scalar("loss/gen", loss_gen.item(), step)
                writer.add_scalar("loss/disc", loss_disc.item(), step)

                if use_research and dis_loss is not None:
                    writer.add_scalar("loss/disentangle", loss_dis_total.item(), step)

                if mine_mi != 0.0:
                    writer.add_scalar("loss/mine_mi", mine_mi, step)

                writer.add_scalar("grad/grad_norm_g", grad_norm_g, step)
                writer.add_scalar("grad/grad_norm_d", grad_norm_d, step)
                writer.add_scalar("lr/learning_rate", lr, step)

                # Log attention weights (research only)
                if use_research and attn_w is not None:
                    timbre_mass = attn_w[..., 0].mean().item()
                    writer.add_scalar("fusion/timbre_attn_mass", timbre_mass, step)
                    writer.add_scalar("fusion/prosody_attn_mass", 1.0 - timbre_mass, step)

            # ================================================================
            # Checkpoint Saving
            # ================================================================
            if step % hps.train.eval_interval == 0 and step > 0:
                # Save generator
                ckpt_g_path = os.path.join(args.model_dir, f"G_{step}.pth")
                save_checkpoint(net_g, optimizer_g, hps.train.learning_rate, step, ckpt_g_path)

                # Save discriminator
                ckpt_d_path = os.path.join(args.model_dir, f"D_{step}.pth")
                save_checkpoint(net_d, optimizer_d, hps.train.learning_rate, step, ckpt_d_path)

                print(f"Saved checkpoints at step {step}")

            step += 1
            if step >= total_steps:
                break

        if step >= total_steps:
            break

    print("Training complete!")
    writer.close()


if __name__ == "__main__":
    train()

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self, *args, **kwargs): pass
from utils import get_hparams_from_file, save_checkpoint, load_checkpoint
from models_research import SynthesizerTrnResearch
from disentangle_loss import build_disentangle
from data_utils_research import TextAudioSpeakerLoaderResearch, TextAudioSpeakerCollateResearch

# Set random seed
torch.manual_seed(1234)
np.random.seed(1234)

class MockFullDataset(Dataset):
    """
    Mock dataset for train_full.py execution check.
    Returns shapes that match standard VITS + Research outputs.
    """
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random length T_text and T_spec
        T_text = np.random.randint(20, 80)
        T_spec = T_text * 2 + np.random.randint(-5, 5)
        T_spec = max(10, T_spec)
        
        text = torch.randint(1, 20, (T_text,), dtype=torch.long)
        spec = torch.randn(80, T_spec, dtype=torch.float32)
        wav = torch.randn(T_spec * 256, dtype=torch.float32) # hop_length = 256
        sid = torch.tensor(np.random.randint(0, 109), dtype=torch.long)
        
        p_feat = torch.randn(T_text, 3, dtype=torch.float32)
        voiced = torch.ByteTensor((torch.rand(T_text) > 0.4).byte())
        p_feat[voiced == 0, 0] = 0.0 # set unvoiced pitch to 0
        p_mask = torch.ones(T_text, dtype=torch.uint8)
        
        g_timbre = torch.randn(192, dtype=torch.float32)
        
        return (text, spec, wav, sid, p_feat, p_mask, voiced, g_timbre)

def main():
    parser = argparse.ArgumentParser(description="Joint training (VITS + fusion + disentanglement)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_dir", type=str, default="checkpoints/full", help="Directory to save checkpoints")
    parser.add_argument("--pretrain_prosody_pth", type=str, default=None, help="Pretrained prosody encoder weights path")
    parser.add_argument("--force_mock", action="store_true", help="Force using mock dataset")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    hps = get_hparams_from_file(args.config)
    
    # Summary writer
    writer = SummaryWriter(log_dir=os.path.join(args.model_dir, "logs"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build dataset and loaders
    use_mock = args.force_mock or not os.path.exists(hps.data.training_files)
    if use_mock:
        print("Using MOCK dataset for training.")
        train_dataset = MockFullDataset(size=200)
        val_dataset = MockFullDataset(size=20)
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
        num_workers=0, # 0 for Windows compatibility
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=hps.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Initialize SynthesizerTrnResearch
    net_g = SynthesizerTrnResearch(
        n_vocab=200, # vocab size
        spec_channels=80,
        segment_size=hps.train.segment_size,
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

    # Load pre-trained prosody encoder weights if available
    if args.pretrain_prosody_pth is not None and os.path.exists(args.pretrain_prosody_pth):
        print(f"Initializing ProsodyEncoder from checkpoint: {args.pretrain_prosody_pth}")
        checkpoint = torch.load(args.pretrain_prosody_pth, map_location=device)
        # Check if saved model has 'model' key containing state dict
        state_dict = checkpoint.get("model", checkpoint)
        net_g.prosody_enc.load_state_dict(state_dict, strict=True)
    else:
        print("ProsodyEncoder initialized with random weights.")

    # Disentangle loss module setup
    dis_loss = build_disentangle(hps.research)
    if dis_loss is not None:
        dis_loss = dis_loss.to(device)
        
    # Standard VITS generator optimizer
    optimizer_g = torch.optim.AdamW(
        net_g.parameters(),
        lr=hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )
    
    # MINE specific optimizer setup
    optimizer_mine = None
    if hps.research.disentangle_loss == "mine" and dis_loss is not None:
        print("Initializing statistics network optimizer for MINE.")
        optimizer_mine = torch.optim.Adam(
            dis_loss.T.parameters(),
            lr=1e-4,
            betas=(0.5, 0.9)
        )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=hps.train.fp16_run)

    print("Starting joint training loop...")
    step = 0
    total_steps = 100000 if not use_mock else 100
    
    for epoch in range(1000):
        net_g.train()
        for batch in train_loader:
            # Unpack collate output
            (x_padded, x_lengths, spec_padded, spec_lengths,
             wav_padded, wav_lengths, sid,
             p_feat_padded, p_mask_padded, voiced_padded, g_timbre_padded) = [
                 b.to(device) if torch.is_tensor(b) else b for b in batch
             ]
            
            # 1. Update statistics network (for MINE) if enabled
            # Guard: skip step 0 — statistics net needs at least one gradient before use
            if hps.research.disentangle_loss == "mine" and dis_loss is not None and step > 0:
                # Get embeddings from net_g in eval/no-grad mode for discriminator update stability
                net_g.eval()
                with torch.no_grad():
                    p_emb = net_g.prosody_enc(p_feat_padded, p_mask_padded)
                net_g.train()
                
                # Perform multiple discriminator updates per generator step (k = 3)
                mine_mi = 0.0
                for _ in range(3):
                    mine_mi = dis_loss.update_statistics_net(g_timbre_padded, p_emb, p_mask_padded, optimizer_mine)
                writer.add_scalar("loss/mine_mi_bound", mine_mi, step)

            # 2. Generator Step
            optimizer_g.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=hps.train.fp16_run):
                outputs, extras = net_g(
                    x=x_padded,
                    x_lengths=x_lengths,
                    spec=spec_padded,
                    spec_lengths=spec_lengths,
                    g_timbre=g_timbre_padded,
                    p_feat=p_feat_padded,
                    p_mask=p_mask_padded,
                    sid=sid
                )
                
                # Unpack baseline outputs
                o, logw_, z, y_mask, x_mask, stats_t = outputs
                m_p, logs_p, m_q, logs_q = stats_t
                
                # Unpack research extras
                g_t, p, p_mask, attn_w = extras
                
                # Baseline VITS losses (Dummy version representing generator losses)
                # In real VITS, this would be a combination of reconstruction, dur, flow, GAN losses
                # We calculate standard regression loss terms for compiling/mock purposes
                loss_recon = F.l1_loss(o, torch.zeros_like(o)) # Mel reconstruction placeholder
                loss_flow = torch.mean(z**2)                   # Flow loss placeholder
                loss_dur = torch.mean(logw_**2)                # Duration loss placeholder
                
                loss_vits_generator = loss_recon + 0.1 * loss_flow + 0.1 * loss_dur
                
                # Add explicit disentanglement loss penalty
                loss_dis = torch.tensor(0.0, device=device)
                if dis_loss is not None:
                    # Cosine or MINE penalty
                    loss_dis = dis_loss(g_t, p, p_mask)
                    
                loss_total = loss_vits_generator + hps.research.disentangle_weight * loss_dis

            scaler.scale(loss_total).backward()
            # Gradient clipping
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(net_g.parameters(), 1.0)
            scaler.step(optimizer_g)
            scaler.update()

            # Logging
            if step % hps.train.log_interval == 0:
                print(f"Step {step} | Total Loss: {loss_total.item():.4f} | VITS Gen Loss: {loss_vits_generator.item():.4f} | Dis Loss: {loss_dis.item():.4f}")
                writer.add_scalar("loss/total", loss_total.item(), step)
                writer.add_scalar("loss/vits_generator", loss_vits_generator.item(), step)
                writer.add_scalar("loss/disentanglement", loss_dis.item(), step)
                
                # Log attention weights metrics (diagnostics)
                if attn_w is not None:
                    # attn_w shape: [B, T, 1+T]
                    # First column is attention mass on timbre token
                    timbre_mass = attn_w[..., 0].mean().item()
                    writer.add_scalar("fusion/timbre_attn_mass", timbre_mass, step)
                    writer.add_scalar("fusion/prosody_attn_mass", 1.0 - timbre_mass, step)
                
                # Log projection diagnostics for collapse detector
                if hps.research.disentangle_loss == "cosine" and dis_loss is not None:
                    m = p_mask_padded.unsqueeze(-1).float()
                    p_bar = (p * m).sum(1) / m.sum(1).clamp(min=1.0)
                    zt = F.normalize(dis_loss.wt(g_timbre_padded), dim=-1)
                    zp = F.normalize(dis_loss.wp(p_bar), dim=-1)
                    
                    writer.add_scalar("diagnostics/zt_std", zt.std().item(), step)
                    writer.add_scalar("diagnostics/zp_std", zp.std().item(), step)

            # Checkpoint save
            if step % hps.train.eval_interval == 0 and step > 0:
                ckpt_path = os.path.join(args.model_dir, f"model_step_{step}.pth")
                save_checkpoint(net_g, optimizer_g, hps.train.learning_rate, step, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
                
            step += 1
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    print("Joint training complete.")
    writer.close()

if __name__ == "__main__":
    main()

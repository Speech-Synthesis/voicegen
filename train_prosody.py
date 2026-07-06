import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self, *args, **kwargs): pass
from utils import get_hparams_from_file
from prosody_encoder import ProsodyEncoder, ProsodyRecon

# Set random seed
torch.manual_seed(1234)
np.random.seed(1234)

class ProsodyPretrainDataset(Dataset):
    """
    Dataset for standalone prosody pretraining.
    Loads from the extended filelist or generates mock data if not available.
    """
    def __init__(self, filelist_path, is_mock=False):
        self.is_mock = is_mock
        if not is_mock and os.path.exists(filelist_path):
            self.items = []
            with open(filelist_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 4:
                        self.items.append(parts[3]) # prosody npy path
            if len(self.items) == 0:
                print("Warning: Filelist was empty, falling back to mock mode.")
                self.is_mock = True
        else:
            self.is_mock = True

        if self.is_mock:
            print("Running in MOCK mode. Generating synthetic dataset...")
            self.items = [None] * 500 # 500 mock samples

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.is_mock:
            # Generate a random phoneme length between 10 and 150
            T = np.random.randint(10, 150)
            
            # Generate mock features: Column 0: Pitch, 1: Energy, 2: Duration
            p_feat = np.random.randn(T, 3).astype(np.float32)
            # Voicing mask: about 60% of phonemes are voiced
            voiced = (np.random.rand(T) > 0.4).astype(np.uint8)
            p_feat[voiced == 0, 0] = 0.0 # Set unvoiced pitch to 0
            
            p_mask = np.ones(T, dtype=np.uint8)
            
            return torch.FloatTensor(p_feat), torch.BoolTensor(p_mask), torch.ByteTensor(voiced)
        
        # Real data mode
        npy_path = self.items[idx]
        p_feat = np.load(npy_path)
        voiced_path = npy_path.replace(".npy", ".voiced.npy")
        if os.path.exists(voiced_path):
            voiced = np.load(voiced_path)
        else:
            voiced = (p_feat[:, 0] != 0.0).astype(np.uint8)
            
        p_mask = np.ones(p_feat.shape[0], dtype=np.uint8)
        
        return torch.FloatTensor(p_feat), torch.BoolTensor(p_mask), torch.ByteTensor(voiced)

def collate_pretrain(batch):
    # Sort by sequence length descending
    batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
    max_len = batch[0][0].size(0)
    
    B = len(batch)
    p_feat_padded = torch.zeros(B, max_len, 3)
    p_mask_padded = torch.zeros(B, max_len, dtype=torch.bool)
    voiced_padded = torch.zeros(B, max_len, dtype=torch.uint8)
    
    for i, (feat, mask, voice) in enumerate(batch):
        T = feat.size(0)
        p_feat_padded[i, :T, :] = feat
        p_mask_padded[i, :T] = mask
        voiced_padded[i, :T] = voice
        
    return p_feat_padded, p_mask_padded, voiced_padded

def compute_correlation(x, y):
    """Compute Pearson correlation coefficient."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = torch.sqrt(((x - x_mean)**2).sum() * ((y - y_mean)**2).sum())
    if den == 0:
        return 0.0
    return float(num / den)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    all_pitch_true = []
    all_pitch_pred = []
    all_energy_true = []
    all_energy_pred = []
    
    with torch.no_grad():
        for batch in val_loader:
            p_feat, p_mask, voiced = [b.to(device) for b in batch]
            loss = model.loss(p_feat, p_mask, voiced)
            total_loss += loss.item()
            
            # Predict
            z = model(p_feat, p_mask)
            pred = model.dec(z)
            
            # Extract voiced pitch elements for correlation
            vm = (p_mask & voiced.bool())
            if vm.any():
                all_pitch_true.append(p_feat[..., 0][vm].cpu())
                all_pitch_pred.append(pred[..., 0][vm].cpu())
                
            # Extract all valid energy elements
            if p_mask.any():
                all_energy_true.append(p_feat[..., 1][p_mask].cpu())
                all_energy_pred.append(pred[..., 1][p_mask].cpu())

    mean_loss = total_loss / len(val_loader)
    
    # Calculate correlations
    pitch_corr = 0.0
    energy_corr = 0.0
    if all_pitch_true:
        pitch_corr = compute_correlation(torch.cat(all_pitch_true), torch.cat(all_pitch_pred))
    if all_energy_true:
        energy_corr = compute_correlation(torch.cat(all_energy_true), torch.cat(all_energy_pred))
        
    return mean_loss, pitch_corr, energy_corr

def main():
    parser = argparse.ArgumentParser(description="Standalone pretraining for ProsodyEncoder")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_dir", type=str, default="checkpoints/prosody", help="Directory to save checkpoints")
    parser.add_argument("--force_mock", action="store_true", help="Force using mock dataset")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    hps = get_hparams_from_file(args.config)
    
    # Tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(args.model_dir, "logs"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build dataset & loaders
    train_dataset = ProsodyPretrainDataset(hps.data.training_files, is_mock=args.force_mock)
    val_dataset = ProsodyPretrainDataset(hps.data.validation_files, is_mock=args.force_mock or train_dataset.is_mock)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=0, # keep simple for Windows compatibility
        collate_fn=collate_pretrain
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=hps.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pretrain
    )

    # Initialize modules
    enc = ProsodyEncoder(
        in_dim=3,
        hidden=256,
        out_dim=hps.research.prosody_dim
    )
    model = ProsodyRecon(enc).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    print("Starting pretraining loop...")
    step = 0
    best_pitch_corr = -1.0
    
    # We run for a configurable number of epochs/steps
    # Guide target: ~20k steps. On mock data, we can run fewer steps for validation.
    total_steps = 20000 if not train_dataset.is_mock else 500
    epochs = int(np.ceil(total_steps / len(train_loader)))
    
    print(f"Total planned steps: {total_steps} ({epochs} epochs)")
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            p_feat, p_mask, voiced = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            loss = model.loss(p_feat, p_mask, voiced)
            loss.backward()
            optimizer.step()
            
            writer.add_scalar("loss/train", loss.item(), step)
            
            if step % hps.train.log_interval == 0:
                print(f"Epoch {epoch} | Step {step} | Train Loss: {loss.item():.4f}")
                
            if step % hps.train.eval_interval == 0 or step == total_steps - 1:
                val_loss, pitch_corr, energy_corr = evaluate(model, val_loader, device)
                model.train()
                
                print(f"[VAL] Step {step} | Loss: {val_loss:.4f} | Pitch Corr: {pitch_corr:.4f} | Energy Corr: {energy_corr:.4f}")
                writer.add_scalar("loss/val", val_loss, step)
                writer.add_scalar("metrics/pitch_correlation", pitch_corr, step)
                writer.add_scalar("metrics/energy_correlation", energy_corr, step)
                
                # Check acceptance and save best
                if pitch_corr > best_pitch_corr:
                    best_pitch_corr = pitch_corr
                    ckpt_path = os.path.join(args.model_dir, "best_model.pth")
                    torch.save({
                        "model": enc.state_dict(),
                        "recon_model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "pitch_corr": pitch_corr
                    }, ckpt_path)
                    print(f"Saved best model checkpoint to {ckpt_path} (Pitch Corr: {pitch_corr:.4f})")
                    
            step += 1
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    print(f"Prosody pretraining complete. Best Pitch Corr: {best_pitch_corr:.4f}")
    writer.close()

if __name__ == "__main__":
    main()

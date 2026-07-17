#!/usr/bin/env python3
"""
HPC Training Wrapper for VITS Voice Cloning

Features:
- Comprehensive system monitoring (GPU, CPU, RAM)
- ETA estimation and progress tracking
- Automatic checkpoint resumption
- Clean, formatted console output
- Graceful interrupt handling
- Periodic status reports

Usage:
    python hpc_train.py --config configs/vctk_base.json --model_dir checkpoints/vctk
    python hpc_train.py --config configs/vctk_full.json --model_dir checkpoints/full --auto_resume
"""

import os
import sys
import argparse
import subprocess
import time
import signal
import glob
import re
import threading
from datetime import datetime, timedelta
from collections import deque

# ============================================================================
# System Monitoring Utilities
# ============================================================================

def get_gpu_stats():
    """Get GPU utilization, memory, and temperature using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append({
                        'utilization': int(parts[0]),
                        'memory_used': int(parts[1]),
                        'memory_total': int(parts[2]),
                        'temperature': int(parts[3])
                    })
            return gpus
    except Exception:
        pass
    return []


def get_cpu_usage():
    """Get CPU usage percentage."""
    try:
        # Try psutil first
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except ImportError:
        pass

    # Fallback to /proc/stat on Linux
    try:
        with open('/proc/stat', 'r') as f:
            line = f.readline()
            parts = line.split()
            if parts[0] == 'cpu':
                idle = int(parts[4])
                total = sum(int(p) for p in parts[1:])
                return round((1 - idle / total) * 100, 1)
    except Exception:
        pass

    return -1


def get_ram_usage():
    """Get RAM usage in MB and percentage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'used_mb': mem.used // (1024 * 1024),
            'total_mb': mem.total // (1024 * 1024),
            'percent': mem.percent
        }
    except ImportError:
        pass

    # Fallback to /proc/meminfo on Linux
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
            mem_info = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    mem_info[parts[0].rstrip(':')] = int(parts[1])

            total = mem_info.get('MemTotal', 0)
            available = mem_info.get('MemAvailable', mem_info.get('MemFree', 0))
            used = total - available
            return {
                'used_mb': used // 1024,
                'total_mb': total // 1024,
                'percent': round(used / total * 100, 1) if total > 0 else 0
            }
    except Exception:
        pass

    return {'used_mb': -1, 'total_mb': -1, 'percent': -1}


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 0:
        return "N/A"

    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 24:
        days = hours // 24
        hours = hours % 24
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_number(n):
    """Format large numbers with commas."""
    return f"{n:,}"


# ============================================================================
# Checkpoint Management
# ============================================================================

def find_latest_checkpoint(model_dir):
    """Find the latest generator checkpoint in the model directory."""
    pattern = os.path.join(model_dir, "G_*.pth")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None, 0

    # Extract step numbers and find max
    steps = []
    for ckpt in checkpoints:
        match = re.search(r'G_(\d+)\.pth', ckpt)
        if match:
            steps.append((int(match.group(1)), ckpt))

    if not steps:
        return None, 0

    steps.sort(key=lambda x: x[0], reverse=True)
    return steps[0][1], steps[0][0]


def count_checkpoints(model_dir):
    """Count the number of saved checkpoints."""
    pattern = os.path.join(model_dir, "G_*.pth")
    return len(glob.glob(pattern))


# ============================================================================
# Training Monitor
# ============================================================================

class TrainingMonitor:
    """Monitor training progress and system resources."""

    def __init__(self, total_steps, start_step=0, log_file=None):
        self.total_steps = total_steps
        self.start_step = start_step
        self.current_step = start_step
        self.current_epoch = 0
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.loss_history = deque(maxlen=100)
        self.log_file = log_file
        self.running = True

        # Loss tracking
        self.last_total_loss = 0.0
        self.last_mel_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_lr = 0.0

    def update(self, step, epoch, total_loss, mel_loss=0, kl_loss=0, lr=0):
        """Update training state."""
        self.current_step = step
        self.current_epoch = epoch
        self.last_total_loss = total_loss
        self.last_mel_loss = mel_loss
        self.last_kl_loss = kl_loss
        self.last_lr = lr
        self.loss_history.append(total_loss)

    def get_progress(self):
        """Calculate training progress percentage."""
        if self.total_steps <= 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100

    def get_eta(self):
        """Estimate time remaining."""
        elapsed = time.time() - self.start_time
        steps_done = self.current_step - self.start_step

        if steps_done <= 0:
            return -1

        steps_remaining = self.total_steps - self.current_step
        time_per_step = elapsed / steps_done
        eta_seconds = steps_remaining * time_per_step

        return eta_seconds

    def get_throughput(self):
        """Calculate steps per second."""
        elapsed = time.time() - self.start_time
        steps_done = self.current_step - self.start_step

        if elapsed <= 0:
            return 0.0

        return steps_done / elapsed

    def get_avg_loss(self):
        """Get average loss from recent history."""
        if not self.loss_history:
            return 0.0
        return sum(self.loss_history) / len(self.loss_history)

    def format_status_report(self, model_dir=None):
        """Generate a formatted status report."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time
        progress = self.get_progress()
        eta = self.get_eta()
        throughput = self.get_throughput()

        # System stats
        gpu_stats = get_gpu_stats()
        cpu_usage = get_cpu_usage()
        ram = get_ram_usage()

        # Build report
        lines = [
            "",
            "=" * 70,
            f"  TRAINING STATUS REPORT - {now}",
            "=" * 70,
            "",
            "  PROGRESS:",
            f"    Step:          {format_number(self.current_step)} / {format_number(self.total_steps)}",
            f"    Epoch:         {self.current_epoch}",
            f"    Progress:      {progress:.1f}%",
            f"    Elapsed:       {format_time(elapsed)}",
            f"    ETA:           {format_time(eta)}",
            f"    Throughput:    {throughput:.2f} steps/sec",
            "",
            "  TRAINING METRICS:",
            f"    Total Loss:    {self.last_total_loss:.4f}",
            f"    Mel Loss:      {self.last_mel_loss:.4f}",
            f"    KL Loss:       {self.last_kl_loss:.4f}",
            f"    Avg Loss:      {self.get_avg_loss():.4f} (last 100 steps)",
            f"    Learning Rate: {self.last_lr:.2e}",
            "",
        ]

        # GPU stats
        if gpu_stats:
            lines.append("  GPU STATUS:")
            for i, gpu in enumerate(gpu_stats):
                lines.append(f"    GPU {i}:")
                lines.append(f"      Utilization: {gpu['utilization']}%")
                lines.append(f"      Memory:      {gpu['memory_used']} / {gpu['memory_total']} MB ({gpu['memory_used']*100//gpu['memory_total']}%)")
                lines.append(f"      Temperature: {gpu['temperature']}°C")
            lines.append("")

        # CPU/RAM stats
        lines.extend([
            "  SYSTEM STATUS:",
            f"    CPU Usage:     {cpu_usage}%",
            f"    RAM Usage:     {ram['used_mb']} / {ram['total_mb']} MB ({ram['percent']}%)",
        ])

        # Checkpoint info
        if model_dir:
            num_ckpts = count_checkpoints(model_dir)
            latest_ckpt, latest_step = find_latest_checkpoint(model_dir)
            lines.extend([
                "",
                "  CHECKPOINTS:",
                f"    Saved:         {num_ckpts} checkpoints",
                f"    Latest:        Step {format_number(latest_step)}" if latest_step > 0 else "    Latest:        None",
            ])

        lines.extend([
            "",
            "=" * 70,
            ""
        ])

        return "\n".join(lines)


# ============================================================================
# Training Process Wrapper
# ============================================================================

class TrainingProcess:
    """Wrapper for the training subprocess with monitoring."""

    def __init__(self, config, model_dir, resume_path=None, extra_args=None):
        self.config = config
        self.model_dir = model_dir
        self.resume_path = resume_path
        self.extra_args = extra_args or []
        self.process = None
        self.monitor = None
        self.log_file = None
        self.running = False

    def build_command(self):
        """Build the training command."""
        cmd = [
            sys.executable, "train_full.py",
            "--config", self.config,
            "--model_dir", self.model_dir
        ]

        if self.resume_path:
            cmd.extend(["--resume", self.resume_path])

        cmd.extend(self.extra_args)
        return cmd

    def start(self, log_path=None):
        """Start the training process."""
        cmd = self.build_command()

        print(f"Starting training with command:")
        print(f"  {' '.join(cmd)}")
        print()

        # Open log file if specified
        if log_path:
            self.log_file = open(log_path, 'a', buffering=1)
            self.log_file.write(f"\n{'='*70}\n")
            self.log_file.write(f"Training started at {datetime.now()}\n")
            self.log_file.write(f"Command: {' '.join(cmd)}\n")
            self.log_file.write(f"{'='*70}\n\n")

        # Start subprocess
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        self.running = True
        return self.process

    def stop(self):
        """Stop the training process gracefully."""
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()

        if self.log_file:
            self.log_file.close()


# ============================================================================
# Main Training Wrapper
# ============================================================================

def parse_training_output(line, monitor):
    """Parse training output to extract metrics."""
    # Pattern: "Step 1000 | Epoch 5 | Loss Total: 12.345 | Loss Mel: 1.234 | ..."
    step_match = re.search(r'Step\s+(\d+)', line)
    epoch_match = re.search(r'Epoch\s+(\d+)', line)
    total_loss_match = re.search(r'Loss Total:\s+([\d.]+)', line)
    mel_loss_match = re.search(r'Loss Mel:\s+([\d.]+)', line)
    kl_loss_match = re.search(r'Loss KL:\s+([\d.]+)', line)

    if step_match and total_loss_match:
        step = int(step_match.group(1))
        epoch = int(epoch_match.group(1)) if epoch_match else 0
        total_loss = float(total_loss_match.group(1))
        mel_loss = float(mel_loss_match.group(1)) if mel_loss_match else 0
        kl_loss = float(kl_loss_match.group(1)) if kl_loss_match else 0

        # Extract learning rate if available
        lr_match = re.search(r'lr[:\s]+([\d.e-]+)', line, re.IGNORECASE)
        lr = float(lr_match.group(1)) if lr_match else 2e-4

        monitor.update(step, epoch, total_loss, mel_loss, kl_loss, lr)
        return True

    return False


def run_training(args):
    """Run training with monitoring."""

    # Setup
    os.makedirs(args.model_dir, exist_ok=True)
    log_dir = os.path.join(args.model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"training_{timestamp}.log")

    # Find checkpoint for auto-resume
    resume_path = args.resume
    start_step = 0

    if args.auto_resume and not resume_path:
        latest_ckpt, latest_step = find_latest_checkpoint(args.model_dir)
        if latest_ckpt:
            resume_path = latest_ckpt
            start_step = latest_step
            print(f"Auto-resuming from checkpoint: {latest_ckpt} (step {latest_step})")

    # Get total steps from config
    total_steps = args.total_steps
    if total_steps <= 0:
        # Try to read from config
        try:
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
                total_steps = config.get('train', {}).get('total_steps', 100000)
        except Exception:
            total_steps = 100000

    # Initialize monitor
    monitor = TrainingMonitor(total_steps, start_step, log_path)

    # Build extra args
    extra_args = []
    if args.pretrain_prosody:
        extra_args.extend(["--pretrain_prosody_pth", args.pretrain_prosody])
    if args.force_mock:
        extra_args.append("--force_mock")

    # Start training process
    training = TrainingProcess(args.config, args.model_dir, resume_path, extra_args)
    process = training.start(log_path)

    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n\nReceived interrupt signal. Stopping training gracefully...")
        training.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Print initial status
    print(f"\nTraining started at {datetime.now()}")
    print(f"Config: {args.config}")
    print(f"Model dir: {args.model_dir}")
    print(f"Log file: {log_path}")
    print(f"Total steps: {format_number(total_steps)}")
    if resume_path:
        print(f"Resuming from: {resume_path} (step {start_step})")
    print()
    print("Press Ctrl+C to stop training gracefully.")
    print("-" * 70)
    print()

    # Main monitoring loop
    last_status_time = time.time()
    status_interval = args.status_interval  # seconds

    try:
        with open(log_path, 'a', buffering=1) as log_file:
            for line in process.stdout:
                line = line.rstrip()

                # Print to console
                print(line)

                # Write to log file
                log_file.write(line + "\n")

                # Parse for metrics
                parse_training_output(line, monitor)

                # Periodic status report
                if time.time() - last_status_time >= status_interval:
                    report = monitor.format_status_report(args.model_dir)
                    print(report)
                    log_file.write(report + "\n")
                    last_status_time = time.time()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    finally:
        training.stop()

        # Final status report
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Final step: {format_number(monitor.current_step)}")
        print(f"Total time: {format_time(time.time() - monitor.start_time)}")

        latest_ckpt, latest_step = find_latest_checkpoint(args.model_dir)
        if latest_ckpt:
            print(f"Latest checkpoint: {latest_ckpt}")

        print(f"Log file: {log_path}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="HPC Training Wrapper for VITS Voice Cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python hpc_train.py --config configs/vctk_base.json --model_dir checkpoints/vctk

  # Auto-resume from latest checkpoint
  python hpc_train.py --config configs/vctk_base.json --model_dir checkpoints/vctk --auto_resume

  # With pretrained prosody encoder
  python hpc_train.py --config configs/vctk_full.json --model_dir checkpoints/full \\
      --pretrain_prosody checkpoints/prosody/best_model.pth --auto_resume

  # Custom total steps and status interval
  python hpc_train.py --config configs/vctk_base.json --model_dir checkpoints/vctk \\
      --total_steps 200000 --status_interval 300

SSH Session Management:
  # Using tmux (recommended)
  tmux new -s training
  python hpc_train.py --config configs/vctk_base.json --model_dir checkpoints/vctk --auto_resume 2>&1 | tee -a training.log
  # Press Ctrl+B, then D to detach
  # Later: tmux attach -t training

  # Using nohup (simpler, no session management)
  nohup python hpc_train.py --config configs/vctk_base.json --model_dir checkpoints/vctk --auto_resume > training.log 2>&1 &
"""
    )

    # Required arguments
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config file")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory to save checkpoints")

    # Resume options
    parser.add_argument("--resume", type=str, default=None,
                        help="Explicit checkpoint path to resume from")
    parser.add_argument("--auto_resume", action="store_true",
                        help="Automatically resume from latest checkpoint in model_dir")

    # Training options
    parser.add_argument("--total_steps", type=int, default=-1,
                        help="Total training steps (default: read from config)")
    parser.add_argument("--pretrain_prosody", type=str, default=None,
                        help="Path to pretrained prosody encoder weights")
    parser.add_argument("--force_mock", action="store_true",
                        help="Force using mock dataset for testing")

    # Monitoring options
    parser.add_argument("--status_interval", type=int, default=300,
                        help="Seconds between status reports (default: 300 = 5 minutes)")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    run_training(args)


if __name__ == "__main__":
    main()

#!/bin/bash
# =============================================================================
# HPC Training Launch Script
# =============================================================================
#
# Usage:
#   ./hpc_launch.sh --config configs/vctk_base.json --model_dir checkpoints/vctk
#   ./hpc_launch.sh --config configs/vctk_full.json --model_dir checkpoints/full --auto_resume
#
# This script:
#   1. Detects available session managers (tmux > screen > nohup)
#   2. Creates a named session for training
#   3. Logs all output to a file
#   4. Supports auto-resume from latest checkpoint
#
# =============================================================================

set -e

# Default values
CONFIG=""
MODEL_DIR=""
SESSION_NAME="vits_training"
AUTO_RESUME=""
EXTRA_ARGS=""
LOG_DIR="logs"
FORCE_BACKEND=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --session)
            SESSION_NAME="$2"
            shift 2
            ;;
        --auto_resume)
            AUTO_RESUME="--auto_resume"
            shift
            ;;
        --backend)
            FORCE_BACKEND="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --config <config.json> --model_dir <dir> [options]"
            echo ""
            echo "Options:"
            echo "  --config        Path to training config (required)"
            echo "  --model_dir     Directory for checkpoints (required)"
            echo "  --session       Session name (default: vits_training)"
            echo "  --auto_resume   Auto-resume from latest checkpoint"
            echo "  --backend       Force backend: tmux, screen, or nohup"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIG" || -z "$MODEL_DIR" ]]; then
    echo "Error: --config and --model_dir are required"
    echo "Use --help for usage information"
    exit 1
fi

# Create directories
mkdir -p "$MODEL_DIR"
mkdir -p "$LOG_DIR"

# Timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${SESSION_NAME}_${TIMESTAMP}.log"

# Build training command
TRAIN_CMD="python hpc_train.py --config $CONFIG --model_dir $MODEL_DIR $AUTO_RESUME $EXTRA_ARGS 2>&1 | tee -a $LOG_FILE"

# Detect available session manager
detect_backend() {
    if [[ -n "$FORCE_BACKEND" ]]; then
        echo "$FORCE_BACKEND"
        return
    fi

    if command -v tmux &> /dev/null; then
        echo "tmux"
    elif command -v screen &> /dev/null; then
        echo "screen"
    else
        echo "nohup"
    fi
}

BACKEND=$(detect_backend)

echo "============================================================"
echo "  HPC Training Launch Script"
echo "============================================================"
echo ""
echo "  Config:      $CONFIG"
echo "  Model Dir:   $MODEL_DIR"
echo "  Session:     $SESSION_NAME"
echo "  Backend:     $BACKEND"
echo "  Log File:    $LOG_FILE"
echo "  Auto Resume: ${AUTO_RESUME:-disabled}"
echo ""

# Launch based on backend
case $BACKEND in
    tmux)
        echo "Starting training in tmux session: $SESSION_NAME"
        echo ""

        # Check if session already exists
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "Warning: Session '$SESSION_NAME' already exists!"
            echo "To attach: tmux attach -t $SESSION_NAME"
            echo "To kill:   tmux kill-session -t $SESSION_NAME"
            exit 1
        fi

        # Create new tmux session and run training
        tmux new-session -d -s "$SESSION_NAME" "bash -c '$TRAIN_CMD; echo; echo Training finished. Press Enter to close.; read'"

        echo "============================================================"
        echo "  Training started in tmux session!"
        echo "============================================================"
        echo ""
        echo "  Commands:"
        echo "    Attach to session:    tmux attach -t $SESSION_NAME"
        echo "    Detach from session:  Ctrl+B, then D"
        echo "    List sessions:        tmux ls"
        echo "    Kill session:         tmux kill-session -t $SESSION_NAME"
        echo ""
        echo "  Monitoring:"
        echo "    Watch GPU:            watch -n 1 nvidia-smi"
        echo "    Tail log:             tail -f $LOG_FILE"
        echo "    Check process:        ps aux | grep train"
        echo ""
        echo "  You can safely close your SSH connection now."
        echo "============================================================"
        ;;

    screen)
        echo "Starting training in screen session: $SESSION_NAME"
        echo ""

        # Check if session already exists
        if screen -list | grep -q "$SESSION_NAME"; then
            echo "Warning: Session '$SESSION_NAME' already exists!"
            echo "To attach: screen -r $SESSION_NAME"
            echo "To kill:   screen -X -S $SESSION_NAME quit"
            exit 1
        fi

        # Create new screen session and run training
        screen -dmS "$SESSION_NAME" bash -c "$TRAIN_CMD; echo; echo Training finished. Press Enter to close.; read"

        echo "============================================================"
        echo "  Training started in screen session!"
        echo "============================================================"
        echo ""
        echo "  Commands:"
        echo "    Attach to session:    screen -r $SESSION_NAME"
        echo "    Detach from session:  Ctrl+A, then D"
        echo "    List sessions:        screen -ls"
        echo "    Kill session:         screen -X -S $SESSION_NAME quit"
        echo ""
        echo "  You can safely close your SSH connection now."
        echo "============================================================"
        ;;

    nohup)
        echo "Starting training with nohup (no session management)"
        echo ""

        # Run with nohup
        nohup bash -c "$TRAIN_CMD" > /dev/null 2>&1 &
        PID=$!

        # Save PID for later reference
        echo $PID > "$MODEL_DIR/training.pid"

        echo "============================================================"
        echo "  Training started in background!"
        echo "============================================================"
        echo ""
        echo "  Process ID: $PID"
        echo "  PID saved to: $MODEL_DIR/training.pid"
        echo ""
        echo "  Commands:"
        echo "    Check if running:     ps -p $PID"
        echo "    Tail log:             tail -f $LOG_FILE"
        echo "    Stop training:        kill $PID"
        echo ""
        echo "  You can safely close your SSH connection now."
        echo "============================================================"
        ;;
esac

echo ""
echo "Log file: $LOG_FILE"

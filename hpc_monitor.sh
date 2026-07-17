#!/bin/bash
# =============================================================================
# HPC Training Monitor Script
# =============================================================================
#
# Run this from another terminal to monitor training status.
#
# Usage:
#   ./hpc_monitor.sh                    # Full dashboard
#   ./hpc_monitor.sh --gpu              # GPU only
#   ./hpc_monitor.sh --log              # Tail training log
#   ./hpc_monitor.sh --checkpoints      # List checkpoints
#   ./hpc_monitor.sh --watch            # Continuous monitoring
#
# =============================================================================

MODEL_DIR="${MODEL_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-logs}"
WATCH_MODE=false
MODE="dashboard"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --gpu)
            MODE="gpu"
            shift
            ;;
        --log)
            MODE="log"
            shift
            ;;
        --checkpoints)
            MODE="checkpoints"
            shift
            ;;
        --process)
            MODE="process"
            shift
            ;;
        --watch)
            WATCH_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_dir DIR   Checkpoint directory to monitor"
            echo "  --gpu             Show GPU status only"
            echo "  --log             Tail training log"
            echo "  --checkpoints     List saved checkpoints"
            echo "  --process         Check training process status"
            echo "  --watch           Continuous monitoring mode"
            echo "  --help            Show this help"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

show_gpu_status() {
    print_header "GPU STATUS"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits | while IFS=, read -r idx name util mem_used mem_total temp; do
            echo ""
            echo "  GPU $idx: $name"
            echo "    Utilization:  ${util}%"
            echo "    Memory:       ${mem_used} / ${mem_total} MB"
            echo "    Temperature:  ${temp}°C"
        done
        echo ""
    else
        echo "  nvidia-smi not available"
    fi
}

show_cpu_ram() {
    print_header "CPU & MEMORY"

    # CPU usage
    if command -v mpstat &> /dev/null; then
        CPU_USAGE=$(mpstat 1 1 | awk '/Average:/ {print 100 - $NF}')
        echo "  CPU Usage: ${CPU_USAGE}%"
    elif [[ -f /proc/stat ]]; then
        CPU_LINE=$(head -1 /proc/stat)
        CPU_VALS=($CPU_LINE)
        IDLE=${CPU_VALS[4]}
        TOTAL=0
        for v in "${CPU_VALS[@]:1}"; do
            TOTAL=$((TOTAL + v))
        done
        if [[ $TOTAL -gt 0 ]]; then
            CPU_USAGE=$((100 - (IDLE * 100 / TOTAL)))
            echo "  CPU Usage: ${CPU_USAGE}%"
        fi
    fi

    # RAM usage
    if command -v free &> /dev/null; then
        free -h | awk '/^Mem:/ {print "  RAM Usage: " $3 " / " $2 " (" int($3/$2*100) "%)"}'
    elif [[ -f /proc/meminfo ]]; then
        TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        AVAIL=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
        if [[ -z "$AVAIL" ]]; then
            AVAIL=$(grep MemFree /proc/meminfo | awk '{print $2}')
        fi
        USED=$((TOTAL - AVAIL))
        PCT=$((USED * 100 / TOTAL))
        echo "  RAM Usage: $((USED/1024)) / $((TOTAL/1024)) MB (${PCT}%)"
    fi
    echo ""
}

show_training_process() {
    print_header "TRAINING PROCESS"

    # Check for Python training processes
    PROCS=$(pgrep -af "python.*train" 2>/dev/null || true)

    if [[ -n "$PROCS" ]]; then
        echo "  Active training processes:"
        echo "$PROCS" | while read -r line; do
            echo "    $line"
        done
    else
        echo "  No training processes found"
    fi

    # Check tmux sessions
    if command -v tmux &> /dev/null; then
        echo ""
        echo "  tmux sessions:"
        tmux ls 2>/dev/null | while read -r line; do
            echo "    $line"
        done || echo "    (none)"
    fi

    # Check screen sessions
    if command -v screen &> /dev/null; then
        echo ""
        echo "  screen sessions:"
        screen -ls 2>/dev/null | grep -E "^\s+[0-9]+" | while read -r line; do
            echo "    $line"
        done || echo "    (none)"
    fi

    echo ""
}

show_checkpoints() {
    print_header "CHECKPOINTS"

    # Find all checkpoint directories
    for dir in checkpoints/*/; do
        if [[ -d "$dir" ]]; then
            CKPTS=$(ls -1 "$dir"G_*.pth 2>/dev/null | wc -l)
            if [[ $CKPTS -gt 0 ]]; then
                LATEST=$(ls -1t "$dir"G_*.pth 2>/dev/null | head -1)
                LATEST_STEP=$(echo "$LATEST" | grep -oE '[0-9]+' | tail -1)
                echo "  $dir"
                echo "    Count:  $CKPTS checkpoints"
                echo "    Latest: Step $LATEST_STEP ($(basename "$LATEST"))"
                echo "    Size:   $(du -sh "$dir" | cut -f1)"
                echo ""
            fi
        fi
    done
}

show_logs() {
    print_header "RECENT LOG OUTPUT"

    # Find most recent log file
    LATEST_LOG=$(ls -1t $LOG_DIR/training_*.log 2>/dev/null | head -1)

    if [[ -n "$LATEST_LOG" && -f "$LATEST_LOG" ]]; then
        echo "  Log file: $LATEST_LOG"
        echo "  Last modified: $(stat -c %y "$LATEST_LOG" 2>/dev/null || stat -f %Sm "$LATEST_LOG" 2>/dev/null)"
        echo ""
        echo "  Last 20 lines:"
        echo "  ---"
        tail -20 "$LATEST_LOG" | sed 's/^/  /'
        echo "  ---"
    else
        echo "  No log files found in $LOG_DIR/"
    fi
    echo ""
}

tail_log() {
    LATEST_LOG=$(ls -1t $LOG_DIR/training_*.log 2>/dev/null | head -1)

    if [[ -n "$LATEST_LOG" && -f "$LATEST_LOG" ]]; then
        echo "Tailing $LATEST_LOG (Ctrl+C to stop)..."
        echo ""
        tail -f "$LATEST_LOG"
    else
        echo "No log files found in $LOG_DIR/"
        exit 1
    fi
}

show_dashboard() {
    clear
    echo ""
    echo "============================================================"
    echo "       HPC TRAINING MONITOR - $(date)"
    echo "============================================================"

    show_training_process
    show_gpu_status
    show_cpu_ram
    show_checkpoints
    show_logs
}

# =============================================================================
# Main
# =============================================================================

case $MODE in
    gpu)
        show_gpu_status
        ;;
    log)
        tail_log
        ;;
    checkpoints)
        show_checkpoints
        ;;
    process)
        show_training_process
        ;;
    dashboard)
        if $WATCH_MODE; then
            while true; do
                show_dashboard
                sleep 30
            done
        else
            show_dashboard
        fi
        ;;
esac

#!/bin/bash
# Script to run training with version suffix v4 in the background
# This script ensures training continues even if SSH connection is lost

# Activate virtual environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_v4_${TIMESTAMP}.log"

echo "Starting training with version suffix v4..."
echo "Log file: $LOG_FILE"
echo "To view progress: tail -f $LOG_FILE"
echo ""

# Run training in background with nohup
# Using version_suffix=v4 and random_order=True (default)
# Using very small number of steps per model for testing (but all M values)
nohup python run_m_diversity_experiment.py \
    --version_suffix v4 \
    --max_power 20 \
    --num_steps 5 \
    --batch_size 2048 \
    --learning_rate 1e-3 \
    --warmup_steps 2 \
    --print_every 1 \
    --checkpoint_every 5 \
    > "$LOG_FILE" 2>&1 &

TRAINING_PID=$!

echo "Training started with PID: $TRAINING_PID"
echo "Process is running in the background and will survive SSH disconnection"
echo ""
echo "Useful commands:"
echo "  tail -f $LOG_FILE            # View log file in real-time"
echo "  ps -p $TRAINING_PID          # Check if process is still running"
echo "  kill $TRAINING_PID           # Stop the training (if needed)"
echo ""
echo "PID saved to: logs/training_v4_${TIMESTAMP}.pid"
echo $TRAINING_PID > "logs/training_v4_${TIMESTAMP}.pid"

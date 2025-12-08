#!/bin/bash
# Wrapper script to run the M diversity experiment in a way that survives SSH disconnection
# Usage: ./run_experiment.sh [arguments to run_m_diversity_experiment.py]

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/experiment_${TIMESTAMP}.log"

echo "Starting experiment..."
echo "Log file: $LOG_FILE"
echo "To view progress: tail -f $LOG_FILE"
echo "To reattach to tmux session: tmux attach -t experiment"
echo ""

# Activate virtual environment and run with nohup (survives SSH disconnection)
# Find venv directory
if [ -d "venv" ]; then
    VENV_DIR="venv"
elif [ -d ".venv" ]; then
    VENV_DIR=".venv"
else
    echo "Error: Virtual environment not found. Please create one first."
    exit 1
fi

# Run with nohup, activating venv first
# Use exec to properly handle arguments
nohup bash -c "source $VENV_DIR/bin/activate && python run_m_diversity_experiment.py \"\$@\"" _ "$@" > "$LOG_FILE" 2>&1 &
EXPERIMENT_PID=$!

echo "Experiment started with PID: $EXPERIMENT_PID"
echo "Process is running in the background and will survive SSH disconnection"
echo ""
echo "Useful commands:"
echo "  tail -f $LOG_FILE            # View log file in real-time"
echo "  ps -p $EXPERIMENT_PID        # Check if process is still running"
echo "  kill $EXPERIMENT_PID         # Stop the experiment (if needed)"
echo ""
echo "PID saved to: logs/experiment_${TIMESTAMP}.pid"
echo $EXPERIMENT_PID > "logs/experiment_${TIMESTAMP}.pid"


#!/bin/bash
# Monitor and compare task vectors as training progresses
# This script periodically checks if task vectors match for completed M values

VERSION_SUFFIX="${1:-v4}"
INTERVAL="${2:-30}"  # Check every 30 seconds by default
RECOVERED_DIR="${3:-task_vectors}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/monitor_task_vectors_${VERSION_SUFFIX}_${TIMESTAMP}.log"

echo "Monitoring task vector comparison for version: $VERSION_SUFFIX"
echo "Checking every $INTERVAL seconds"
echo "Recovered task vectors directory: $RECOVERED_DIR"
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "Monitoring stopped at $(date)"
    echo "Log saved to: $LOG_FILE"
    exit 0
}

trap cleanup SIGINT SIGTERM

while true; do
    {
        echo "=========================================="
        echo "$(date): Running comparison..."
        echo "=========================================="
        
        python compare_task_vectors.py \
            --version-suffix "$VERSION_SUFFIX" \
            --recovered-dir "$RECOVERED_DIR"
        
        echo ""
        echo "Next check in $INTERVAL seconds..."
        echo ""
    } | tee -a "$LOG_FILE"
    
    sleep "$INTERVAL"
done

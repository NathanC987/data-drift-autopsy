#!/bin/bash

# Launch Drift Autopsy Dashboard
# Usage: ./launch_dashboard.sh [port]

PORT=${1:-8501}

echo "======================================================================"
echo "  Drift Autopsy Dashboard"
echo "======================================================================"
echo ""
echo "Starting dashboard on port $PORT..."
echo "Dashboard will open in your browser at: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "======================================================================"
echo ""

if [ -x "venv/bin/python" ]; then
	venv/bin/python -m streamlit run examples/dashboard/app.py --server.port "$PORT"
else
	streamlit run examples/dashboard/app.py --server.port "$PORT"
fi

#!/bin/bash
set -e

# Download data if not present
if [ ! -d "data/index" ]; then
    echo "First run — downloading data and building index..."
    bash setup.sh
fi

echo "Starting server..."
exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 app:app

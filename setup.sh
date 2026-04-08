#!/bin/bash
# Setup script for Global Equity Market Return Predictor

set -e  # Exit on error

echo "==============================================="
echo "Global Equity Market Return Predictor - Setup"
echo "==============================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python 3.8+
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "ERROR: Python 3.8+ required, found $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo ""
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "IMPORTANT: Edit .env and add your FRED API key to get full functionality"
    echo "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
else
    echo ""
    echo ".env file already exists"
fi

# Create necessary directories
echo ""
echo "Creating data directories..."
mkdir -p data/{raw,processed,features}
mkdir -p models
mkdir -p reports
mkdir -p logs

# Run tests if pytest is available
echo ""
echo "Running basic tests..."
if python -m pytest tests/ -v --tb=short 2>/dev/null; then
    echo "✓ Tests passed"
else
    echo "⚠ No tests found or tests failed (this is OK for initial setup)"
fi

echo ""
echo "==============================================="
echo "Setup Complete!"
echo "==============================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your FRED API key (optional but recommended)"
echo "  2. Activate the environment: source venv/bin/activate"
echo "  3. Run the full pipeline: python run_prediction.py --mode full"
echo ""
echo "For help: python run_prediction.py --help"
echo ""

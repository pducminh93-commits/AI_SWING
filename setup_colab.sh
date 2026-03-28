#!/bin/bash
# Simple setup script for Google Colab
echo "Setting up the environment for AI Swing Trading..."

# Install all dependencies from requirements.txt
# This now includes pandas-ta and other libraries, which are pure Python
# and do not require complex compilation steps like the old ta-lib.
echo "Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "Setup complete. Please upload your .env file and config directory."

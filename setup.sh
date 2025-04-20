#!/bin/bash

# Financial PDF to Google Sheets Setup Script

echo "Setting up Financial PDF to Google Sheets..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Create a virtual environment
echo "Creating a virtual environment..."
python3 -m venv venv

# Activate the virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    echo "Activating virtual environment for Windows..."
    source venv/Scripts/activate
else
    # Linux/MacOS
    echo "Activating virtual environment for Unix-like OS..."
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for Google API credentials
echo "Checking for Google API credentials..."
if [ ! -f "credentials.json" ] && [ ! -f "service_account.json" ]; then
    echo "Warning: No Google API credentials found."
    echo "Please obtain OAuth credentials (credentials.json) or a service account key (service_account.json)"
    echo "from the Google Cloud Console and place the file in this directory."
fi

# Create .env file from template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit the .env file to add your API keys and other configuration."
fi

# Create uploads directory if it doesn't exist
if [ ! -d "uploads" ]; then
    echo "Creating uploads directory..."
    mkdir uploads
fi

# Make the CLI script executable
chmod +x process_pdf.py

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Add your Google credentials file (credentials.json or service_account.json)"
echo "3. Run the application: python app.py"
echo "4. Open http://localhost:5000 in your browser" 
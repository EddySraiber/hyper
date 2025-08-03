#!/bin/bash

# Algotrading Agent Setup Script

set -e

echo "🚀 Setting up Algotrading Agent..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs config monitoring

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your Alpaca API credentials"
else
    echo "✅ .env file already exists"
fi

# Create data subdirectories
echo "📂 Setting up data directories..."
mkdir -p data/{news_scraper,news_filter,news_analysis_brain,decision_engine,risk_manager,statistical_advisor}

# Set permissions
echo "🔒 Setting permissions..."
chmod -R 755 data logs
chmod +x main.py

# Create docker-compose override for development
if [ ! -f docker-compose.override.yml ]; then
    echo "🐳 Creating docker-compose override for development..."
    cat > docker-compose.override.yml << EOF
version: '3.8'

services:
  algotrading-agent:
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - .:/app:rw  # Mount source code for development
EOF
fi

# Create basic monitoring config
echo "📊 Setting up monitoring configuration..."
mkdir -p monitoring
cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'algotrading-agent'
    static_configs:
      - targets: ['algotrading-agent:8080']
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your Alpaca API credentials"
echo "2. Run: docker-compose up --build"
echo "3. Monitor logs for trading activity"
echo ""
echo "For paper trading, your .env should contain:"
echo "   ALPACA_API_KEY=your_key_here"
echo "   ALPACA_SECRET_KEY=your_secret_here" 
echo "   ALPACA_PAPER_TRADING=true"
echo ""
echo "📚 See README.md for more details"
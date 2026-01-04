#!/bin/bash

# Script to create an independent VibeFrame 2.0 repository
# Run this script from the parent directory of Vibehub-

echo "🚀 Creating independent VibeFrame 2.0 repository..."

# Step 1: Create new directory
echo "📁 Creating new directory..."
mkdir -p ../vibeframe-2.0
cd ../vibeframe-2.0

# Step 2: Initialize git
echo "🔧 Initializing git repository..."
git init

# Step 3: Copy all VibeFrame files
echo "📋 Copying VibeFrame files..."
cp -r ../Vibehub-/ai_mv_generator/* .

# Step 4: Create initial commit
echo "💾 Creating initial commit..."
git add -A
git commit -m "Initial commit: VibeFrame 2.0 - AI Music Video Generator

Complete implementation with:
- Audio analysis with librosa
- Scene planning with LLM
- Video generation (LongCat, SHARP, HunyuanVideo)
- Video composition with FFmpeg/MoviePy
- Project management
- Gradio web interface
- Comprehensive test suite (211 tests)
- Full documentation"

echo "✅ Local repository created!"
echo ""
echo "📝 Next steps:"
echo "1. Create a new repository on GitHub: https://github.com/new"
echo "   - Name it: vibeframe-2.0"
echo "   - Make it public or private"
echo "   - Don't initialize with README (we already have one)"
echo ""
echo "2. Add the remote and push:"
echo "   cd ../vibeframe-2.0"
echo "   git remote add origin https://github.com/Readyyeahgoooo/vibeframe-2.0.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "🎉 Done! Your independent VibeFrame 2.0 repository will be ready!"

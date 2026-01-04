# 🎵 VibeFrame 2.0: AI Music Video Generator

Transform your audio into visually consistent, professionally edited music videos using AI.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🎵 **Advanced Audio Analysis**: Beat detection, tempo analysis, and musical structure recognition using librosa
- 🎬 **AI-Powered Scene Generation**: LLM-based scene planning with OpenRouter integration
- 🤖 **Multiple AI Models**: Support for LongCat-Video, HunyuanVideo, and SHARP 3D generation
- 🎨 **Character Consistency**: Maintain visual consistency across all scenes
- 🔄 **Smart Fallback System**: Automatic model fallback and template-based generation
- 📱 **Platform Presets**: Optimized for YouTube, Instagram, TikTok, and more
- 💰 **Free Tier Optimized**: Works with free API tiers and open-source models
- 🌐 **Web Interface**: User-friendly Gradio interface

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vibeframe-2.0.git
cd vibeframe-2.0

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (optional)
```

### Basic Usage

```bash
# Start the web interface
python app_gradio.py
```

Then open your browser to `http://localhost:7860`

### Command Line Usage

```python
from vibeframe.workflow import WorkflowOrchestrator

# Initialize workflow
workflow = WorkflowOrchestrator(
    openrouter_api_key="your-key-here",  # Optional
    huggingface_token="your-token-here"   # Optional
)

# Generate music video
result = workflow.run_complete_workflow(
    audio_path="path/to/your/music.mp3",
    project_name="my_music_video",
    global_style="cinematic",
    theme="music video",
    resolution="1080p",
    fps=30
)

print(f"Video saved to: {result['final_video_path']}")
```

## 📋 Requirements

- Python 3.9+
- FFmpeg (for video processing)
- 8GB+ RAM recommended
- GPU optional (for faster video generation)

### System Dependencies

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## 🎯 Platform Presets

VibeFrame 2.0 includes optimized presets for popular platforms:

```python
from vibeframe.config import get_config

config = get_config()

# YouTube preset (1080p, 30fps, 16:9)
youtube_settings = config.get_platform_preset("youtube")

# Instagram preset (1080x1080, 30fps, 1:1)
instagram_settings = config.get_platform_preset("instagram")

# TikTok preset (1080x1920, 30fps, 9:16)
tiktok_settings = config.get_platform_preset("tiktok")
```

Available presets:
- `youtube` - 1920x1080, 30fps, 16:9
- `instagram` - 1080x1080, 30fps, 1:1
- `instagram_story` - 1080x1920, 30fps, 9:16
- `tiktok` - 1080x1920, 30fps, 9:16
- `twitter` - 1280x720, 30fps, 16:9
- `facebook` - 1280x720, 30fps, 16:9

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Keys (Optional - system works without them)
OPENROUTER_API_KEY=sk-or-...
HUGGINGFACE_TOKEN=hf_...

# Model Settings
DEFAULT_MODEL=longcat
DEFAULT_RESOLUTION=1920x1080
DEFAULT_FPS=30

# Cache Settings
CACHE_DIR=cache
ENABLE_CACHING=true
```

### API Keys

**OpenRouter** (Optional):
- Provides better scene descriptions via LLM
- Get your key at [openrouter.ai](https://openrouter.ai)
- Falls back to templates if not provided

**HuggingFace** (Optional):
- Higher rate limits for model inference
- Get your token at [huggingface.co](https://huggingface.co/settings/tokens)
- Works without token (lower rate limits)

## 📖 Usage Guide

### 1. Audio Analysis

```python
from vibeframe.audio_analyzer import AudioAnalyzer

analyzer = AudioAnalyzer()
result = analyzer.analyze_audio("music.mp3")

print(f"Duration: {result['duration']}s")
print(f"Tempo: {result['features']['tempo']} BPM")
print(f"Cut points: {len(result['cut_points'])}")
```

### 2. Scene Planning

```python
from vibeframe.scene_planner import ScenePlanner

planner = ScenePlanner(api_key="your-openrouter-key")
scenes = planner.generate_scene_descriptions(
    cut_points=result['cut_points'],
    audio_features=result['features'],
    global_style="cinematic",
    theme="music video"
)
```

### 3. Video Generation

```python
from vibeframe.video_generator import VideoGenerator

generator = VideoGenerator(hf_token="your-hf-token")
video_path = generator.generate_text_to_video(
    prompt="cinematic shot of a musician performing",
    duration=5.0,
    resolution=(1920, 1080),
    fps=24
)
```

### 4. Video Assembly

```python
from vibeframe.video_compositor import VideoCompositor

compositor = VideoCompositor()

# Normalize clips
normalized = compositor.normalize_clips(
    clip_paths=["clip1.mp4", "clip2.mp4"],
    target_resolution=(1920, 1080),
    target_fps=30
)

# Concatenate with transitions
final = compositor.concatenate_clips(
    normalized,
    transition_type="fade",
    transition_duration=0.5
)

# Sync with audio
synced = compositor.synchronize_audio(
    final,
    "music.mp3",
    sync_method="trim"
)

# Export
compositor.export_final_video(
    synced,
    "output.mp4",
    quality="high"
)
```

## 🎨 Customization

### Custom Styles

```python
# Define custom visual style
custom_style = {
    "color_palette": "warm tones, golden hour",
    "camera_movement": "smooth tracking shots",
    "lighting": "dramatic side lighting",
    "mood": "energetic and uplifting"
}

scenes = planner.generate_scene_descriptions(
    cut_points=cut_points,
    audio_features=features,
    global_style=custom_style,
    theme="performance video"
)
```

### Character Consistency

```python
from vibeframe.character_manager import CharacterManager

char_manager = CharacterManager()

# Define character
character = "a young woman with long dark hair, wearing a leather jacket"

# Inject into all scenes
for scene in scenes:
    scene.video_prompt = char_manager.inject_character(
        scene.video_prompt,
        character
    )
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_audio_analyzer.py

# Run with coverage
pytest --cov=vibeframe tests/

# Run property-based tests (longer)
pytest tests/ -v --hypothesis-show-statistics
```

## 📊 Project Structure

```
vibeframe-2.0/
├── vibeframe/              # Main package
│   ├── audio_analyzer.py   # Audio analysis
│   ├── scene_planner.py    # Scene generation
│   ├── character_manager.py # Character consistency
│   ├── video_generator.py  # Video generation
│   ├── video_compositor.py # Video assembly
│   ├── project_manager.py  # Project management
│   ├── workflow.py         # End-to-end orchestration
│   ├── config.py           # Configuration
│   ├── error_handler.py    # Error handling
│   ├── api_clients.py      # API clients with rate limiting
│   ├── models.py           # Data models
│   ├── exceptions.py       # Custom exceptions
│   └── utils.py            # Utilities
├── tests/                  # Test suite
├── app_gradio.py          # Web interface
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
└── README.md             # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [librosa](https://librosa.org/) for audio analysis
- [MoviePy](https://zulko.github.io/moviepy/) for video processing
- [Gradio](https://gradio.app/) for the web interface
- [OpenRouter](https://openrouter.ai/) for LLM access
- [HuggingFace](https://huggingface.co/) for model hosting

## 🐛 Troubleshooting

### Common Issues

**"FFmpeg not found"**
```bash
# Install FFmpeg
brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Ubuntu
```

**"CUDA out of memory"**
```python
# Use CPU mode or reduce resolution
generator = VideoGenerator(device="cpu")
# Or reduce resolution
video = generator.generate_text_to_video(
    prompt=prompt,
    resolution=(720, 480)  # Lower resolution
)
```

**"API rate limit exceeded"**
- Wait a few minutes before retrying
- System will automatically use template-based fallback
- Consider upgrading API plan for higher limits

**"Audio file not supported"**
```bash
# Convert to WAV format
ffmpeg -i input.mp3 -ar 44100 -ac 2 output.wav
```

## 📧 Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/yourusername/vibeframe-2.0/issues)
- Check the [documentation](https://github.com/yourusername/vibeframe-2.0/wiki)

## 🗺️ Roadmap

- [ ] Real-time preview during generation
- [ ] More video generation models
- [ ] Advanced camera control
- [ ] Batch processing
- [ ] Cloud deployment options
- [ ] Mobile app

---

Made with ❤️ by the VibeFrame team

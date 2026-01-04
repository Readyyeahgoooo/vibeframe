"""
VibeFrame 2.0 - Gradio Web Interface Entry Point

This is the main entry point for the VibeFrame 2.0 web application.
"""

import gradio as gr
import os
import json
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple, Any

from vibeframe.config import get_config
from vibeframe.workflow import WorkflowOrchestrator
from vibeframe.models import Storyboard
from vibeframe.exceptions import WorkflowError
from vibeframe.utils import setup_logging

# Set up logging
logger = setup_logging()

# Global workflow orchestrator
workflow = None

def initialize_workflow(openrouter_key: str = "", hf_token: str = "") -> WorkflowOrchestrator:
    """Initialize workflow orchestrator with API keys."""
    global workflow
    
    # Use provided keys or fall back to environment variables
    openrouter_api_key = openrouter_key.strip() if openrouter_key.strip() else os.getenv("OPENROUTER_API_KEY")
    huggingface_token = hf_token.strip() if hf_token.strip() else os.getenv("HUGGINGFACE_TOKEN")
    
    workflow = WorkflowOrchestrator(
        openrouter_api_key=openrouter_api_key,
        huggingface_token=huggingface_token
    )
    
    return workflow

def analyze_audio_handler(audio_file: Optional[str], 
                         openrouter_key: str = "", 
                         hf_token: str = "") -> Tuple[str, str]:
    """Handle audio analysis request."""
    try:
        if not audio_file:
            return "", "❌ Please upload an audio file first."
        
        # Initialize workflow with API keys
        wf = initialize_workflow(openrouter_key, hf_token)
        
        # Generate project name from audio file
        audio_path = Path(audio_file)
        project_name = audio_path.stem
        
        logger.info(f"Starting audio analysis for: {project_name}")
        
        # Analyze audio
        result = wf.analyze_audio(audio_file, project_name)
        
        # Generate initial storyboard
        storyboard_result = wf.generate_storyboard(
            project_name=project_name,
            global_style="cinematic",
            theme="music video"
        )
        
        storyboard_json = storyboard_result["storyboard_json"]
        
        status_msg = f"✅ Audio analyzed successfully!\n"
        status_msg += f"📊 Duration: {result['analysis']['duration']:.1f}s\n"
        status_msg += f"🎵 Tempo: {result['analysis']['features']['tempo']:.1f} BPM\n"
        status_msg += f"🎬 Generated {len(json.loads(storyboard_json)['scenes'])} scenes\n"
        status_msg += f"📝 Project: {project_name}"
        
        return storyboard_json, status_msg
        
    except Exception as e:
        error_msg = f"❌ Audio analysis failed: {str(e)}"
        logger.error(f"Audio analysis error: {traceback.format_exc()}")
        return "", error_msg

def generate_video_handler(storyboard_json: str,
                          model_choice: str,
                          resolution: str,
                          openrouter_key: str = "",
                          hf_token: str = "") -> Tuple[Optional[str], str]:
    """Handle video generation request."""
    try:
        if not storyboard_json.strip():
            return None, "❌ Please analyze audio first to generate a storyboard."
        
        # Parse storyboard to get project name
        try:
            storyboard_data = json.loads(storyboard_json)
            project_name = storyboard_data.get("project_name", "unknown")
        except json.JSONDecodeError:
            return None, "❌ Invalid storyboard JSON. Please analyze audio again."
        
        # Initialize workflow
        wf = initialize_workflow(openrouter_key, hf_token)
        
        logger.info(f"Starting video generation for project: {project_name}")
        
        # Progress tracking
        progress_messages = []
        
        def progress_callback(message: str, progress: float):
            progress_messages.append(f"[{progress:.1f}%] {message}")
            logger.info(f"Progress: {message} ({progress:.1f}%)")
        
        # Generate videos
        video_result = wf.generate_video(
            project_name=project_name,
            model_name=model_choice,
            progress_callback=progress_callback
        )
        
        # Assemble final video
        final_result = wf.assemble_final_video(
            project_name=project_name,
            resolution=resolution,
            fps=30,
            quality="standard",
            progress_callback=progress_callback
        )
        
        video_path = final_result["video_path"]
        
        status_msg = f"✅ Video generation completed!\n"
        status_msg += f"🎬 Generated {video_result['successful_scenes']}/{video_result['total_scenes']} scenes\n"
        status_msg += f"📹 Resolution: {resolution}\n"
        status_msg += f"🎯 Model: {model_choice}\n"
        status_msg += f"📁 Saved: {Path(video_path).name}"
        
        return video_path, status_msg
        
    except Exception as e:
        error_msg = f"❌ Video generation failed: {str(e)}"
        logger.error(f"Video generation error: {traceback.format_exc()}")
        return None, error_msg

def update_storyboard_handler(storyboard_json: str) -> str:
    """Handle storyboard updates."""
    try:
        if not storyboard_json.strip():
            return "❌ No storyboard to validate."
        
        # Validate JSON
        storyboard_data = json.loads(storyboard_json)
        
        # Basic validation
        required_fields = ["project_name", "scenes", "audio_duration"]
        for field in required_fields:
            if field not in storyboard_data:
                return f"❌ Missing required field: {field}"
        
        scenes = storyboard_data.get("scenes", [])
        if not scenes:
            return "❌ Storyboard must have at least one scene."
        
        return f"✅ Storyboard valid: {len(scenes)} scenes"
        
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {str(e)}"
    except Exception as e:
        return f"❌ Validation error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="VibeFrame 2.0: AI Music Video Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎵 VibeFrame 2.0: AI Music Video Generator")
    gr.Markdown("Transform your audio into visually consistent, professionally edited music videos using AI.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📤 Step 1: Upload Audio")
            audio_input = gr.Audio(
                type="filepath", 
                label="Upload Music File",
                format="wav"
            )
            
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            gr.Markdown("## ✏️ Step 2: Edit Storyboard")
            storyboard_editor = gr.Code(
                language="json", 
                label="Storyboard (JSON)", 
                lines=12,
                placeholder="Storyboard will appear here after audio analysis...",
                interactive=True
            )
            
            validate_btn = gr.Button("✅ Validate Storyboard", variant="secondary")
            validation_output = gr.Textbox(
                label="Validation Status",
                lines=2,
                interactive=False
            )
            
            gr.Markdown("## ⚙️ Step 3: Configure Generation")
            with gr.Accordion("🔑 API Keys & Settings", open=False):
                gr.Markdown("""
                **Optional API Keys for Enhanced Features:**
                - **OpenRouter**: Better scene descriptions and planning
                - **HuggingFace**: Higher rate limits for video generation
                """)
                
                openrouter_key = gr.Textbox(
                    label="OpenRouter API Key", 
                    type="password",
                    placeholder="sk-or-... (optional)"
                )
                hf_token = gr.Textbox(
                    label="HuggingFace Token", 
                    type="password",
                    placeholder="hf_... (optional)"
                )
                
                with gr.Row():
                    model_choice = gr.Dropdown(
                        choices=["longcat", "sharp"],
                        value="longcat",
                        label="🎬 Video Generation Model",
                        info="LongCat: Fast text-to-video | SHARP: 3D camera effects"
                    )
                    resolution = gr.Dropdown(
                        choices=["720p", "1080p"],
                        value="1080p",
                        label="📺 Resolution",
                        info="Higher resolution = longer processing time"
                    )
            
            generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Status & Output")
            status_output = gr.Textbox(
                label="Status", 
                lines=8,
                value="🚀 Ready to analyze audio!\n\n📝 Instructions:\n1. Upload a music file (MP3, WAV, etc.)\n2. Click 'Analyze Audio' to generate storyboard\n3. Optionally edit the storyboard JSON\n4. Configure settings and generate video",
                interactive=False
            )
            
            video_output = gr.Video(
                label="Generated Music Video",
                height=400
            )
            
            gr.Markdown("## 📋 Project Info")
            with gr.Accordion("ℹ️ About VibeFrame 2.0", open=False):
                gr.Markdown("""
                ### 🌟 Features
                - 🎵 **Advanced Audio Analysis**: Beat detection, tempo analysis, structure recognition
                - 🎬 **Character Consistency**: Maintain visual consistency across scenes
                - 🤖 **Multiple AI Models**: LongCat-Video, SHARP 3D generation
                - 💰 **Free Tier Optimized**: Works with free API tiers
                - 📱 **Platform Ready**: Optimized for YouTube, Instagram, TikTok
                
                ### 🎯 AI Models
                - **LongCat-Video**: Text-to-video generation with temporal consistency
                - **SHARP**: 2D-to-3D conversion with dynamic camera movements
                
                ### 🔧 Technical Details
                - **Audio Processing**: librosa for advanced audio analysis
                - **Scene Planning**: LLM-powered scene generation with fallback templates
                - **Video Assembly**: FFmpeg-based professional video editing
                - **Quality Control**: Automatic resolution/FPS normalization
                
                ### 📊 Current Status
                🚧 **Beta Version** - Core functionality implemented
                """)

    # Event handlers
    analyze_btn.click(
        analyze_audio_handler,
        inputs=[audio_input, openrouter_key, hf_token],
        outputs=[storyboard_editor, status_output]
    )
    
    generate_btn.click(
        generate_video_handler,
        inputs=[storyboard_editor, model_choice, resolution, openrouter_key, hf_token],
        outputs=[video_output, status_output]
    )
    
    validate_btn.click(
        update_storyboard_handler,
        inputs=[storyboard_editor],
        outputs=[validation_output]
    )
    
    clear_btn.click(
        lambda: ("", "", None, "🚀 Ready for new audio file!"),
        outputs=[storyboard_editor, validation_output, video_output, status_output]
    )

if __name__ == "__main__":
    config = get_config()
    logger.info("Starting VibeFrame 2.0 web interface...")
    logger.info(f"Configuration loaded: {len(config)} settings")
    
    # Initialize with environment variables if available
    initialize_workflow()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
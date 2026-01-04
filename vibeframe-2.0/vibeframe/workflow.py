"""Workflow orchestration for VibeFrame 2.0."""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging

from .audio_analyzer import AudioAnalyzer
from .scene_planner import ScenePlanner
from .character_manager import CharacterManager
from .video_generator import VideoGenerator
from .video_compositor import VideoCompositor
from .project_manager import ProjectManager
from .models import Storyboard, ProcessingStep
from .exceptions import WorkflowError, AudioAnalysisError, ScenePlanningError
from .utils import setup_logging

logger = setup_logging()

class WorkflowOrchestrator:
    """
    Orchestrates the complete VibeFrame 2.0 workflow.
    
    This class manages the end-to-end process from audio analysis
    to final video generation and assembly.
    """
    
    def __init__(self, 
                 openrouter_api_key: Optional[str] = None,
                 huggingface_token: Optional[str] = None,
                 projects_root: Optional[str] = None):
        """
        Initialize workflow orchestrator.
        
        Args:
            openrouter_api_key: API key for OpenRouter (optional)
            huggingface_token: HuggingFace token (optional)
            projects_root: Root directory for projects (optional)
        """
        self.openrouter_api_key = openrouter_api_key
        self.huggingface_token = huggingface_token
        
        # Initialize components
        self.audio_analyzer = AudioAnalyzer()
        self.scene_planner = ScenePlanner(api_key=openrouter_api_key)
        self.character_manager = CharacterManager()
        self.video_generator = VideoGenerator(hf_token=huggingface_token)
        self.video_compositor = VideoCompositor()
        self.project_manager = ProjectManager(projects_root)
        
        logger.info("WorkflowOrchestrator initialized")
    
    def analyze_audio(self, audio_path: str, project_name: str) -> Dict[str, Any]:
        """
        Analyze audio file and create project.
        
        Args:
            audio_path: Path to audio file
            project_name: Name for the project
            
        Returns:
            Dictionary with analysis results and project info
            
        Raises:
            WorkflowError: If analysis fails
        """
        try:
            logger.info(f"Starting audio analysis for: {audio_path}")
            
            # Create project
            project_info = self.project_manager.create_project(audio_path, project_name)
            
            # Analyze audio
            analysis_result = self.audio_analyzer.analyze_audio(audio_path)
            
            # Update project status
            self.project_manager._update_project_metadata(project_name, {
                "audio_analysis": analysis_result,
                "status": ProcessingStep.ANALYSIS.value,
                "progress_percent": 10.0
            })
            
            logger.info(f"Audio analysis completed: {len(analysis_result['cut_points'])} cut points found")
            
            return {
                "project_info": project_info,
                "analysis": analysis_result,
                "status": "Audio analysis completed successfully"
            }
            
        except Exception as e:
            error_msg = f"Audio analysis failed: {str(e)}"
            logger.error(error_msg)
            raise WorkflowError(error_msg) from e
    
    def generate_storyboard(self, 
                          project_name: str, 
                          global_style: str = "cinematic",
                          theme: str = "music video",
                          character_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate storyboard from audio analysis.
        
        Args:
            project_name: Name of the project
            global_style: Global visual style
            theme: Video theme
            character_description: Optional character description
            
        Returns:
            Dictionary with storyboard and status
            
        Raises:
            WorkflowError: If storyboard generation fails
        """
        try:
            logger.info(f"Generating storyboard for project: {project_name}")
            
            # Load project metadata
            project_info = self.project_manager._load_project_metadata(project_name)
            analysis_result = project_info.get("audio_analysis")
            
            if not analysis_result:
                raise WorkflowError("No audio analysis found. Run analyze_audio first.")
            
            # Generate scene descriptions
            scenes = self.scene_planner.generate_scene_descriptions(
                cut_points=analysis_result["cut_points"],
                audio_features=analysis_result["features"],
                global_style=global_style,
                theme=theme
            )
            
            # Process character consistency if character provided
            if character_description:
                for scene in scenes:
                    enhanced_prompt = self.character_manager.inject_character(
                        scene.video_prompt, character_description
                    )
                    scene.video_prompt = enhanced_prompt
                    scene.character_descriptor = self.character_manager.extract_character_description(
                        character_description
                    )
            
            # Create storyboard
            storyboard = Storyboard(
                project_name=project_name,
                audio_file=project_info["audio_file"],
                audio_duration=analysis_result["duration"],
                global_style=global_style,
                theme=theme,
                scenes=scenes
            )
            
            # Save storyboard
            self.project_manager.save_storyboard(project_name, storyboard)
            
            logger.info(f"Storyboard generated: {len(scenes)} scenes")
            
            return {
                "storyboard": storyboard,
                "storyboard_json": storyboard.to_json(),
                "status": f"Storyboard generated with {len(scenes)} scenes"
            }
            
        except Exception as e:
            error_msg = f"Storyboard generation failed: {str(e)}"
            logger.error(error_msg)
            raise WorkflowError(error_msg) from e
    
    def generate_video(self, 
                      project_name: str,
                      model_name: str = "longcat",
                      progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """
        Generate video clips from storyboard.
        
        Args:
            project_name: Name of the project
            model_name: Video generation model to use
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with generation results
            
        Raises:
            WorkflowError: If video generation fails
        """
        try:
            logger.info(f"Starting video generation for project: {project_name}")
            
            # Load storyboard
            storyboard = self.project_manager.load_storyboard(project_name)
            
            generated_clips = []
            total_scenes = len(storyboard.scenes)
            
            for i, scene in enumerate(storyboard.scenes):
                if progress_callback:
                    progress = (i / total_scenes) * 100
                    progress_callback(f"Generating scene {i+1}/{total_scenes}", progress)
                
                logger.info(f"Generating scene {scene.id}: {scene.description}")
                
                try:
                    # Generate video clip
                    if model_name == "longcat":
                        clip_path = self.video_generator.generate_text_to_video(
                            prompt=scene.video_prompt,
                            duration=scene.duration,
                            resolution=(1024, 576),  # Default resolution
                            fps=24
                        )
                    elif model_name == "sharp" and scene.character_descriptor:
                        # Use SHARP for 3D generation if character available
                        clip_path = self.video_generator.generate_with_sharp(
                            prompt=scene.video_prompt,
                            duration=scene.duration,
                            camera_path=None  # Could be enhanced with camera control
                        )
                    else:
                        # Fallback to LongCat
                        clip_path = self.video_generator.generate_text_to_video(
                            prompt=scene.video_prompt,
                            duration=scene.duration,
                            resolution=(1024, 576),
                            fps=24
                        )
                    
                    # Save clip to project
                    project_clip_path = self.project_manager.save_generated_clip(
                        project_name, scene.id, clip_path
                    )
                    
                    generated_clips.append({
                        "scene_id": scene.id,
                        "clip_path": project_clip_path,
                        "duration": scene.duration
                    })
                    
                    logger.info(f"Scene {scene.id} generated successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to generate scene {scene.id}: {str(e)}")
                    # Continue with other scenes
                    continue
            
            if progress_callback:
                progress_callback("Video generation completed", 100.0)
            
            logger.info(f"Video generation completed: {len(generated_clips)}/{total_scenes} scenes")
            
            return {
                "generated_clips": generated_clips,
                "total_scenes": total_scenes,
                "successful_scenes": len(generated_clips),
                "status": f"Generated {len(generated_clips)}/{total_scenes} video clips"
            }
            
        except Exception as e:
            error_msg = f"Video generation failed: {str(e)}"
            logger.error(error_msg)
            raise WorkflowError(error_msg) from e
    
    def assemble_final_video(self, 
                           project_name: str,
                           resolution: str = "1080p",
                           fps: int = 30,
                           quality: str = "standard",
                           progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """
        Assemble final video from generated clips.
        
        Args:
            project_name: Name of the project
            resolution: Target resolution (720p, 1080p, 4K)
            fps: Target frame rate
            quality: Quality preset
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with final video info
            
        Raises:
            WorkflowError: If assembly fails
        """
        try:
            logger.info(f"Assembling final video for project: {project_name}")
            
            if progress_callback:
                progress_callback("Loading project data", 10.0)
            
            # Load project info and storyboard
            project_info = self.project_manager._load_project_metadata(project_name)
            storyboard = self.project_manager.load_storyboard(project_name)
            
            # Get generated clips
            project_dir = self.project_manager._get_project_dir(project_name)
            clips_dir = project_dir / "clips"
            
            clip_paths = []
            for scene in storyboard.scenes:
                clip_filename = f"scene_{scene.id:03d}.mp4"
                clip_path = clips_dir / clip_filename
                if clip_path.exists():
                    clip_paths.append(str(clip_path))
                else:
                    logger.warning(f"Missing clip for scene {scene.id}")
            
            if not clip_paths:
                raise WorkflowError("No video clips found. Generate videos first.")
            
            if progress_callback:
                progress_callback("Normalizing clips", 30.0)
            
            # Parse resolution
            resolution_map = {
                "720p": (1280, 720),
                "1080p": (1920, 1080),
                "4K": (3840, 2160)
            }
            target_resolution = resolution_map.get(resolution, (1920, 1080))
            
            # Normalize clips
            normalized_clips = self.video_compositor.normalize_clips(
                clip_paths, target_resolution, fps
            )
            
            if progress_callback:
                progress_callback("Concatenating clips", 60.0)
            
            # Concatenate clips
            concatenated_video = self.video_compositor.concatenate_clips(
                normalized_clips, transition_type="cut"
            )
            
            if progress_callback:
                progress_callback("Synchronizing audio", 80.0)
            
            # Synchronize with original audio
            audio_file = project_info["audio_file"]
            synced_video = self.video_compositor.synchronize_audio(
                concatenated_video, audio_file, sync_method="trim"
            )
            
            if progress_callback:
                progress_callback("Exporting final video", 90.0)
            
            # Export final video
            temp_output = tempfile.mktemp(suffix=".mp4")
            final_video = self.video_compositor.export_final_video(
                synced_video, temp_output, quality=quality
            )
            
            # Save to project
            project_video_path = self.project_manager.save_final_video(
                project_name, final_video
            )
            
            if progress_callback:
                progress_callback("Video assembly completed", 100.0)
            
            logger.info(f"Final video assembled: {project_video_path}")
            
            return {
                "video_path": project_video_path,
                "resolution": resolution,
                "fps": fps,
                "quality": quality,
                "clips_used": len(clip_paths),
                "status": "Final video assembled successfully"
            }
            
        except Exception as e:
            error_msg = f"Video assembly failed: {str(e)}"
            logger.error(error_msg)
            raise WorkflowError(error_msg) from e
    
    def run_complete_workflow(self,
                            audio_path: str,
                            project_name: str,
                            global_style: str = "cinematic",
                            theme: str = "music video",
                            character_description: Optional[str] = None,
                            model_name: str = "longcat",
                            resolution: str = "1080p",
                            fps: int = 30,
                            quality: str = "standard",
                            progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """
        Run the complete workflow from audio to final video.
        
        Args:
            audio_path: Path to audio file
            project_name: Name for the project
            global_style: Global visual style
            theme: Video theme
            character_description: Optional character description
            model_name: Video generation model
            resolution: Target resolution
            fps: Target frame rate
            quality: Quality preset
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with complete workflow results
            
        Raises:
            WorkflowError: If any step fails
        """
        try:
            logger.info(f"Starting complete workflow for: {project_name}")
            
            results = {}
            
            # Step 1: Analyze audio
            if progress_callback:
                progress_callback("Analyzing audio", 5.0)
            
            audio_result = self.analyze_audio(audio_path, project_name)
            results["audio_analysis"] = audio_result
            
            # Step 2: Generate storyboard
            if progress_callback:
                progress_callback("Generating storyboard", 15.0)
            
            storyboard_result = self.generate_storyboard(
                project_name, global_style, theme, character_description
            )
            results["storyboard"] = storyboard_result
            
            # Step 3: Generate videos
            if progress_callback:
                progress_callback("Generating video clips", 25.0)
            
            def video_progress(msg, progress):
                if progress_callback:
                    # Map video generation progress to 25-75% range
                    mapped_progress = 25.0 + (progress / 100.0) * 50.0
                    progress_callback(msg, mapped_progress)
            
            video_result = self.generate_video(
                project_name, model_name, video_progress
            )
            results["video_generation"] = video_result
            
            # Step 4: Assemble final video
            if progress_callback:
                progress_callback("Assembling final video", 75.0)
            
            def assembly_progress(msg, progress):
                if progress_callback:
                    # Map assembly progress to 75-100% range
                    mapped_progress = 75.0 + (progress / 100.0) * 25.0
                    progress_callback(msg, mapped_progress)
            
            final_result = self.assemble_final_video(
                project_name, resolution, fps, quality, assembly_progress
            )
            results["final_video"] = final_result
            
            logger.info(f"Complete workflow finished for: {project_name}")
            
            return {
                "success": True,
                "project_name": project_name,
                "final_video_path": final_result["video_path"],
                "results": results,
                "status": "Complete workflow finished successfully"
            }
            
        except Exception as e:
            error_msg = f"Complete workflow failed: {str(e)}"
            logger.error(error_msg)
            raise WorkflowError(error_msg) from e
    
    def get_project_status(self, project_name: str) -> Dict[str, Any]:
        """
        Get current status of a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary with project status
        """
        try:
            status = self.project_manager.get_project_status(project_name)
            return {
                "project_name": status.project_name,
                "current_step": status.current_step,
                "progress_percent": status.progress_percent,
                "total_scenes": status.total_scenes,
                "scenes_generated": status.scenes_generated,
                "created_at": status.created_at.isoformat(),
                "last_modified": status.last_modified.isoformat(),
                "errors": status.errors
            }
        except Exception as e:
            logger.error(f"Failed to get project status: {str(e)}")
            return {"error": str(e)}
    
    def list_projects(self) -> Dict[str, Any]:
        """
        List all projects.
        
        Returns:
            Dictionary with projects list
        """
        try:
            projects = self.project_manager.list_projects()
            return {
                "projects": projects,
                "count": len(projects)
            }
        except Exception as e:
            logger.error(f"Failed to list projects: {str(e)}")
            return {"error": str(e), "projects": [], "count": 0}
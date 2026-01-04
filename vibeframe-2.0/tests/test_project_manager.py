"""Tests for ProjectManager component."""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from vibeframe.project_manager import ProjectManager
from vibeframe.models import (
    Storyboard, SceneDescription, CharacterDescriptor, ProcessingStep, ProjectStatus
)
from vibeframe.exceptions import ProjectError, StoryboardValidationError


class TestProjectManager:
    """Unit tests for ProjectManager."""
    
    @pytest.fixture
    def temp_projects_dir(self):
        """Create temporary projects directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def project_manager(self, temp_projects_dir):
        """Create ProjectManager instance."""
        return ProjectManager(temp_projects_dir)
    
    @pytest.fixture
    def sample_audio_file(self, temp_projects_dir):
        """Create sample audio file."""
        audio_path = Path(temp_projects_dir) / "sample.mp3"
        audio_path.write_bytes(b"fake audio data")
        return str(audio_path)
    
    @pytest.fixture
    def sample_storyboard(self):
        """Create sample storyboard."""
        character = CharacterDescriptor(
            appearance="young woman with long black hair",
            clothing="red leather jacket",
            style="anime style"
        )
        
        scene = SceneDescription(
            id=1,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            description="Opening scene",
            character_action="walking",
            camera_angle="medium shot",
            lighting="natural",
            environment="city street",
            video_prompt="A young woman walking down a city street",
            character_descriptor=character
        )
        
        return Storyboard(
            project_name="Test Project",
            audio_file="test.mp3",
            audio_duration=30.0,
            global_style="anime",
            theme="urban",
            scenes=[scene]
        )
    
    def test_init(self, temp_projects_dir):
        """Test ProjectManager initialization."""
        pm = ProjectManager(temp_projects_dir)
        assert pm.projects_root == Path(temp_projects_dir)
        assert pm.projects_root.exists()
    
    def test_init_default_path(self):
        """Test ProjectManager with default path."""
        pm = ProjectManager()
        expected_path = Path.home() / "VibeFrame" / "projects"
        assert pm.projects_root == expected_path
    
    def test_create_project_success(self, project_manager, sample_audio_file):
        """Test successful project creation."""
        result = project_manager.create_project(sample_audio_file, "Test Project")
        
        assert "project_dir" in result
        assert "project_info" in result
        assert "audio_path" in result
        
        project_dir = Path(result["project_dir"])
        assert project_dir.exists()
        
        # Check directory structure
        assert (project_dir / "scenes").exists()
        assert (project_dir / "clips").exists()
        assert (project_dir / "final").exists()
        assert (project_dir / "temp").exists()
        assert (project_dir / "audio").exists()
        assert (project_dir / "project.json").exists()
        
        # Check audio file was copied
        audio_path = Path(result["audio_path"])
        assert audio_path.exists()
        assert audio_path.read_bytes() == b"fake audio data"
    
    def test_create_project_audio_not_found(self, project_manager):
        """Test project creation with non-existent audio file."""
        with pytest.raises(ProjectError, match="Audio file not found"):
            project_manager.create_project("nonexistent.mp3", "Test Project")
    
    def test_create_project_already_exists(self, project_manager, sample_audio_file):
        """Test project creation when project already exists."""
        project_manager.create_project(sample_audio_file, "Test Project")
        
        with pytest.raises(ProjectError, match="already exists"):
            project_manager.create_project(sample_audio_file, "Test Project")
    
    def test_save_storyboard_success(self, project_manager, sample_audio_file, sample_storyboard):
        """Test successful storyboard saving."""
        project_manager.create_project(sample_audio_file, "Test Project")
        project_manager.save_storyboard("Test Project", sample_storyboard)
        
        # Check storyboard file was created
        project_dir = project_manager._get_project_dir("Test Project")
        storyboard_path = project_dir / "storyboard.json"
        assert storyboard_path.exists()
        
        # Check metadata was updated
        metadata = project_manager._load_project_metadata("Test Project")
        assert metadata["total_scenes"] == 1
        assert metadata["status"] == ProcessingStep.PLANNING.value
        assert metadata["progress_percent"] == 25.0
    
    def test_save_storyboard_project_not_found(self, project_manager, sample_storyboard):
        """Test saving storyboard to non-existent project."""
        with pytest.raises(ProjectError, match="not found"):
            project_manager.save_storyboard("Nonexistent", sample_storyboard)
    
    def test_load_storyboard_success(self, project_manager, sample_audio_file, sample_storyboard):
        """Test successful storyboard loading."""
        project_manager.create_project(sample_audio_file, "Test Project")
        project_manager.save_storyboard("Test Project", sample_storyboard)
        
        loaded_storyboard = project_manager.load_storyboard("Test Project")
        
        assert loaded_storyboard.project_name == sample_storyboard.project_name
        assert len(loaded_storyboard.scenes) == len(sample_storyboard.scenes)
        assert loaded_storyboard.scenes[0].id == sample_storyboard.scenes[0].id
    
    def test_load_storyboard_not_found(self, project_manager, sample_audio_file):
        """Test loading storyboard when file doesn't exist."""
        project_manager.create_project(sample_audio_file, "Test Project")
        
        with pytest.raises(ProjectError, match="Storyboard not found"):
            project_manager.load_storyboard("Test Project")
    
    def test_save_generated_clip(self, project_manager, sample_audio_file, temp_projects_dir):
        """Test saving generated clip."""
        project_manager.create_project(sample_audio_file, "Test Project")
        
        # Create fake clip file
        clip_path = Path(temp_projects_dir) / "test_clip.mp4"
        clip_path.write_bytes(b"fake video data")
        
        result_path = project_manager.save_generated_clip("Test Project", 1, str(clip_path))
        
        # Check clip was saved
        assert Path(result_path).exists()
        assert Path(result_path).read_bytes() == b"fake video data"
        assert "scene_001.mp4" in result_path
        
        # Check metadata was updated
        metadata = project_manager._load_project_metadata("Test Project")
        assert metadata["scenes_generated"] == 1
        assert metadata["status"] == ProcessingStep.GENERATION.value
    
    def test_save_generated_clip_not_found(self, project_manager, sample_audio_file):
        """Test saving non-existent clip."""
        project_manager.create_project(sample_audio_file, "Test Project")
        
        with pytest.raises(ProjectError, match="Clip file not found"):
            project_manager.save_generated_clip("Test Project", 1, "nonexistent.mp4")
    
    def test_get_project_status(self, project_manager, sample_audio_file):
        """Test getting project status."""
        result = project_manager.create_project(sample_audio_file, "Test Project")
        status = project_manager.get_project_status("Test Project")
        
        assert isinstance(status, ProjectStatus)
        assert status.project_name == "Test Project"
        assert status.total_scenes == 0
        assert status.scenes_generated == 0
        assert status.current_step == ProcessingStep.ANALYSIS.value
        assert status.progress_percent == 0.0
    
    def test_cleanup_project(self, project_manager, sample_audio_file, temp_projects_dir):
        """Test project cleanup."""
        project_manager.create_project(sample_audio_file, "Test Project")
        
        # Add some files to temp and clips directories
        project_dir = project_manager._get_project_dir("Test Project")
        temp_file = project_dir / "temp" / "temp.txt"
        temp_file.write_text("temp data")
        
        clips_file = project_dir / "clips" / "clip.mp4"
        clips_file.write_bytes(b"clip data")
        
        # Create final video
        final_file = project_dir / "final" / "final.mp4"
        final_file.write_bytes(b"final video")
        
        project_manager.cleanup_project("Test Project", keep_final_video=True)
        
        # Check temp files were cleaned
        assert not temp_file.exists()
        # Check clips were cleaned (final video exists)
        assert not clips_file.exists()
        # Check final video still exists
        assert final_file.exists()
    
    def test_list_projects(self, project_manager, sample_audio_file):
        """Test listing projects."""
        # Create multiple projects
        project_manager.create_project(sample_audio_file, "Project 1")
        project_manager.create_project(sample_audio_file, "Project 2")
        
        projects = project_manager.list_projects()
        
        assert len(projects) == 2
        project_names = [p["name"] for p in projects]
        assert "Project 1" in project_names
        assert "Project 2" in project_names
        
        # Check required fields
        for project in projects:
            assert "name" in project
            assert "created_at" in project
            assert "project_dir" in project
            assert "size_mb" in project
    
    def test_delete_project(self, project_manager, sample_audio_file):
        """Test project deletion."""
        project_manager.create_project(sample_audio_file, "Test Project")
        project_dir = project_manager._get_project_dir("Test Project")
        
        assert project_dir.exists()
        
        project_manager.delete_project("Test Project")
        
        assert not project_dir.exists()
    
    def test_save_final_video(self, project_manager, sample_audio_file, temp_projects_dir):
        """Test saving final video."""
        project_manager.create_project(sample_audio_file, "Test Project")
        
        # Create fake final video
        video_path = Path(temp_projects_dir) / "final.mp4"
        video_path.write_bytes(b"final video data")
        
        result_path = project_manager.save_final_video("Test Project", str(video_path))
        
        # Check video was saved
        assert Path(result_path).exists()
        assert Path(result_path).read_bytes() == b"final video data"
        
        # Check metadata was updated
        metadata = project_manager._load_project_metadata("Test Project")
        assert metadata["status"] == ProcessingStep.COMPLETE.value
        assert metadata["progress_percent"] == 100.0
        assert "final_video" in metadata
    
    def test_update_scene_in_storyboard(self, project_manager, sample_audio_file, sample_storyboard):
        """Test updating scene in storyboard."""
        project_manager.create_project(sample_audio_file, "Test Project")
        project_manager.save_storyboard("Test Project", sample_storyboard)
        
        updates = {
            "description": "Updated description",
            "camera_angle": "close-up"
        }
        
        project_manager.update_scene_in_storyboard("Test Project", 1, updates)
        
        # Load and check updated storyboard
        updated_storyboard = project_manager.load_storyboard("Test Project")
        scene = updated_storyboard.scenes[0]
        
        assert scene.description == "Updated description"
        assert scene.camera_angle == "close-up"
    
    def test_update_scene_not_found(self, project_manager, sample_audio_file, sample_storyboard):
        """Test updating non-existent scene."""
        project_manager.create_project(sample_audio_file, "Test Project")
        project_manager.save_storyboard("Test Project", sample_storyboard)
        
        with pytest.raises(ProjectError, match="Scene 999 not found"):
            project_manager.update_scene_in_storyboard("Test Project", 999, {"description": "test"})
    
    def test_sanitize_filename(self, project_manager):
        """Test filename sanitization."""
        # Test invalid characters - the regex replaces consecutive invalid chars with single underscore
        assert project_manager._sanitize_filename("test<>:\"/\\|?*") == "test_"
        
        # Test length limit
        long_name = "a" * 150
        sanitized = project_manager._sanitize_filename(long_name)
        assert len(sanitized) <= 100
        
        # Test empty string
        assert project_manager._sanitize_filename("") == "untitled"
        assert project_manager._sanitize_filename("   ") == "untitled"
    
    def test_validate_storyboard_empty_scenes(self, project_manager):
        """Test storyboard validation with empty scenes."""
        # Create storyboard with empty scenes - this should be caught by validation, not model creation
        # We need to bypass the model validation to test the ProjectManager validation
        storyboard = Storyboard.__new__(Storyboard)  # Create without calling __init__
        storyboard.project_name = "Test"
        storyboard.audio_file = "test.mp3"
        storyboard.audio_duration = 30.0
        storyboard.global_style = "anime"
        storyboard.theme = "urban"
        storyboard.scenes = []
        
        with pytest.raises(StoryboardValidationError, match="at least one scene"):
            project_manager._validate_storyboard(storyboard)
    
    def test_validate_storyboard_invalid_timing(self, project_manager):
        """Test storyboard validation with invalid timing."""
        # Create scene with invalid timing - bypass model validation
        scene = SceneDescription.__new__(SceneDescription)
        scene.id = 1
        scene.start_time = 10.0  # Start after end
        scene.end_time = 5.0
        scene.duration = 5.0
        scene.description = "Invalid scene"
        scene.character_action = "walking"
        scene.camera_angle = "medium shot"
        scene.lighting = "natural"
        scene.environment = "city street"
        scene.video_prompt = "Test prompt"
        scene.character_descriptor = None
        
        storyboard = Storyboard.__new__(Storyboard)
        storyboard.project_name = "Test"
        storyboard.audio_file = "test.mp3"
        storyboard.audio_duration = 30.0
        storyboard.global_style = "anime"
        storyboard.theme = "urban"
        storyboard.scenes = [scene]
        
        with pytest.raises(StoryboardValidationError, match="start_time must be before end_time"):
            project_manager._validate_storyboard(storyboard)


# Property-based tests
class TestProjectManagerProperties:
    """Property-based tests for ProjectManager."""
    
    @pytest.fixture
    def temp_projects_dir(self):
        """Create temporary projects directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def project_manager(self, temp_projects_dir):
        """Create ProjectManager instance."""
        return ProjectManager(temp_projects_dir)
    
    @given(st.text(min_size=1, max_size=50).filter(lambda x: x.strip() and x.isascii() and all(c not in '<>:"/\\|?*' for c in x)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_project_creation_round_trip(self, project_name):
        """Property: Created projects can be listed and accessed."""
        import tempfile
        import shutil
        import uuid
        
        # Create isolated temp directory for this test
        temp_dir = tempfile.mkdtemp()
        try:
            pm = ProjectManager(temp_dir)
            
            # Create sample audio file
            audio_path = Path(temp_dir) / "sample.mp3"
            audio_path.write_bytes(b"fake audio data")
            
            # Make project name unique to avoid conflicts
            unique_project_name = f"{project_name}_{uuid.uuid4().hex[:8]}"
            
            # Create project
            result = pm.create_project(str(audio_path), unique_project_name)
            
            # Project should appear in listing
            projects = pm.list_projects()
            project_names = [p["name"] for p in projects]
            assert unique_project_name in project_names
            
            # Project status should be accessible
            status = pm.get_project_status(unique_project_name)
            assert status.project_name == unique_project_name
            
        finally:
            shutil.rmtree(temp_dir)
    
    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=100),  # scene_id
                st.floats(min_value=0, max_value=100),    # start_time
                st.floats(min_value=0.1, max_value=10)    # duration
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_storyboard_persistence_round_trip(self, scene_data):
        """Property: Storyboards can be saved and loaded without data loss."""
        import tempfile
        import shutil
        import uuid
        
        # Create isolated temp directory for this test
        temp_dir = tempfile.mkdtemp()
        try:
            pm = ProjectManager(temp_dir)
            
            # Create project with unique name
            audio_path = Path(temp_dir) / "sample.mp3"
            audio_path.write_bytes(b"fake audio data")
            project_name = f"Test_Project_{uuid.uuid4().hex[:8]}"
            pm.create_project(str(audio_path), project_name)
            
            # Create storyboard with generated scenes
            scenes = []
            for i, (scene_id, start_time, duration) in enumerate(scene_data):
                scene = SceneDescription(
                    id=scene_id,
                    start_time=start_time,
                    end_time=start_time + duration,
                    duration=duration,
                    description=f"Scene {i}",
                    character_action="action",
                    camera_angle="medium",
                    lighting="natural",
                    environment="environment",
                    video_prompt=f"Prompt {i}"
                )
                scenes.append(scene)
            
            original_storyboard = Storyboard(
                project_name=project_name,
                audio_file="test.mp3",
                audio_duration=sum(s.duration for s in scenes),
                global_style="anime",
                theme="urban",
                scenes=scenes
            )
            
            # Save and load storyboard
            pm.save_storyboard(project_name, original_storyboard)
            loaded_storyboard = pm.load_storyboard(project_name)
            
            # Verify data integrity
            assert loaded_storyboard.project_name == original_storyboard.project_name
            assert loaded_storyboard.audio_duration == original_storyboard.audio_duration
            assert len(loaded_storyboard.scenes) == len(original_storyboard.scenes)
            
            for orig_scene, loaded_scene in zip(original_storyboard.scenes, loaded_storyboard.scenes):
                assert loaded_scene.id == orig_scene.id
                assert loaded_scene.start_time == orig_scene.start_time
                assert loaded_scene.end_time == orig_scene.end_time
                assert loaded_scene.duration == orig_scene.duration
                
        finally:
            shutil.rmtree(temp_dir)
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_project_listing_completeness(
        self, num_projects
    ):
        """Property: All created projects appear in project listing."""
        import uuid
        import tempfile
        import shutil
        
        # Create isolated temp directory for this test
        temp_dir = tempfile.mkdtemp()
        try:
            pm = ProjectManager(temp_dir)
            
            # Create sample audio file
            audio_path = Path(temp_dir) / "sample.mp3"
            audio_path.write_bytes(b"fake audio data")
            
            # Create multiple projects with unique names
            created_projects = []
            for i in range(num_projects):
                project_name = f"Project_{i}_{uuid.uuid4().hex[:8]}"
                pm.create_project(str(audio_path), project_name)
                created_projects.append(project_name)
            
            # List projects
            projects = pm.list_projects()
            listed_names = [p["name"] for p in projects]
            
            # All created projects should be listed
            for project_name in created_projects:
                assert project_name in listed_names
            
            # Number should match
            assert len(projects) == num_projects
            
        finally:
            shutil.rmtree(temp_dir)
    
    @given(st.text(min_size=1, max_size=50).filter(lambda x: x.strip() and x.isascii() and all(c not in '<>:"/\\|?*' for c in x)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_project_deletion_cleanup(
        self, project_name
    ):
        """Property: Deleted projects are completely removed."""
        import uuid
        import tempfile
        import shutil
        
        # Create isolated temp directory for this test
        temp_dir = tempfile.mkdtemp()
        try:
            pm = ProjectManager(temp_dir)
            
            # Create sample audio file
            audio_path = Path(temp_dir) / "sample.mp3"
            audio_path.write_bytes(b"fake audio data")
            
            # Make project name unique
            unique_project_name = f"{project_name}_{uuid.uuid4().hex[:8]}"
            
            # Create project
            result = pm.create_project(str(audio_path), unique_project_name)
            project_dir = Path(result["project_dir"])
            
            # Verify project exists
            assert project_dir.exists()
            projects_before = pm.list_projects()
            assert any(p["name"] == unique_project_name for p in projects_before)
            
            # Delete project
            pm.delete_project(unique_project_name)
            
            # Verify complete removal
            assert not project_dir.exists()
            projects_after = pm.list_projects()
            assert not any(p["name"] == unique_project_name for p in projects_after)
            
        finally:
            shutil.rmtree(temp_dir)
    
    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=100),  # scene_id
                st.floats(min_value=0, max_value=100),    # start_time
                st.floats(min_value=0.1, max_value=10),   # duration
                st.text(min_size=1, max_size=100)         # description
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_storyboard_json_validity(
        self, scene_data
    ):
        """Property 10: Storyboard JSON Validity - Storyboards serialize to valid JSON."""
        import uuid
        import tempfile
        import shutil
        
        # Create isolated temp directory for this test
        temp_dir = tempfile.mkdtemp()
        try:
            pm = ProjectManager(temp_dir)
            
            # Create project with unique name
            audio_path = Path(temp_dir) / "sample.mp3"
            audio_path.write_bytes(b"fake audio data")
            project_name = f"Test_Project_{uuid.uuid4().hex[:8]}"
            pm.create_project(str(audio_path), project_name)
            
            # Create storyboard with generated scenes
            scenes = []
            for i, (scene_id, start_time, duration, description) in enumerate(scene_data):
                scene = SceneDescription(
                    id=scene_id,
                    start_time=start_time,
                    end_time=start_time + duration,
                    duration=duration,
                    description=description,
                    character_action="action",
                    camera_angle="medium",
                    lighting="natural",
                    environment="environment",
                    video_prompt=f"Prompt {i}"
                )
                scenes.append(scene)
            
            storyboard = Storyboard(
                project_name=project_name,
                audio_file="test.mp3",
                audio_duration=sum(s.duration for s in scenes),
                global_style="anime",
                theme="urban",
                scenes=scenes
            )
            
            # Save storyboard
            pm.save_storyboard(project_name, storyboard)
            
            # Verify JSON is valid by loading it directly
            project_dir = pm._get_project_dir(project_name)
            storyboard_path = project_dir / "storyboard.json"
            
            with open(storyboard_path, 'r') as f:
                json_data = json.load(f)  # Should not raise exception
            
            # Verify essential fields are present
            assert "project_name" in json_data
            assert "scenes" in json_data
            assert isinstance(json_data["scenes"], list)
            assert len(json_data["scenes"]) == len(scenes)
            
        finally:
            shutil.rmtree(temp_dir)
    
    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=100),  # scene_id
                st.floats(min_value=0, max_value=100),    # start_time
                st.floats(min_value=0.1, max_value=10)    # duration
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_storyboard_validation(
        self, scene_data
    ):
        """Property 11: Storyboard Validation - Invalid storyboards are rejected."""
        import uuid
        import tempfile
        import shutil
        
        # Create isolated temp directory for this test
        temp_dir = tempfile.mkdtemp()
        try:
            pm = ProjectManager(temp_dir)
            
            # Create project with unique name
            audio_path = Path(temp_dir) / "sample.mp3"
            audio_path.write_bytes(b"fake audio data")
            project_name = f"Test_Project_{uuid.uuid4().hex[:8]}"
            pm.create_project(str(audio_path), project_name)
            
            # Create valid storyboard first
            valid_scenes = []
            for i, (scene_id, start_time, duration) in enumerate(scene_data):
                scene = SceneDescription(
                    id=scene_id,
                    start_time=start_time,
                    end_time=start_time + duration,
                    duration=duration,
                    description=f"Scene {i}",
                    character_action="action",
                    camera_angle="medium",
                    lighting="natural",
                    environment="environment",
                    video_prompt=f"Prompt {i}"
                )
                valid_scenes.append(scene)
            
            valid_storyboard = Storyboard(
                project_name=project_name,
                audio_file="test.mp3",
                audio_duration=sum(s.duration for s in valid_scenes),
                global_style="anime",
                theme="urban",
                scenes=valid_scenes
            )
            
            # Valid storyboard should save successfully
            pm.save_storyboard(project_name, valid_storyboard)
            
            # Test ProjectManager validation by creating an invalid storyboard that bypasses model validation
            # We'll create a storyboard with scenes that have invalid timing relationships
            invalid_scene = SceneDescription(
                id=1,
                start_time=10.0,  # Start after end - this should be caught by ProjectManager validation
                end_time=5.0,
                duration=5.0,
                description="Invalid scene",
                character_action="action",
                camera_angle="medium",
                lighting="natural",
                environment="environment",
                video_prompt="Invalid prompt"
            )
            
            # Create storyboard with invalid scene timing
            invalid_storyboard = Storyboard(
                project_name=project_name,
                audio_file="test.mp3",
                audio_duration=30.0,
                global_style="anime",
                theme="urban",
                scenes=[invalid_scene]  # This has invalid timing but model won't catch it
            )
            
            # Invalid storyboard should be rejected by ProjectManager validation
            with pytest.raises(StoryboardValidationError):
                pm.save_storyboard(project_name, invalid_storyboard)
                
        finally:
            shutil.rmtree(temp_dir)
    
    @given(
        st.dictionaries(
            st.sampled_from(["description", "camera_angle", "lighting", "environment"]),
            st.text(min_size=1, max_size=50),
            min_size=1,
            max_size=4
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_metadata_preservation_during_edits(
        self, updates
    ):
        """Property 12: Metadata Preservation During Edits - Scene updates preserve other metadata."""
        import uuid
        import tempfile
        import shutil
        
        # Create isolated temp directory for this test
        temp_dir = tempfile.mkdtemp()
        try:
            pm = ProjectManager(temp_dir)
            
            # Create project with unique name
            audio_path = Path(temp_dir) / "sample.mp3"
            audio_path.write_bytes(b"fake audio data")
            project_name = f"Test_Project_{uuid.uuid4().hex[:8]}"
            pm.create_project(str(audio_path), project_name)
            
            # Create storyboard
            original_scene = SceneDescription(
                id=1,
                start_time=0.0,
                end_time=5.0,
                duration=5.0,
                description="Original description",
                character_action="original action",
                camera_angle="original angle",
                lighting="original lighting",
                environment="original environment",
                video_prompt="Original prompt"
            )
            
            storyboard = Storyboard(
                project_name=project_name,
                audio_file="test.mp3",
                audio_duration=5.0,
                global_style="anime",
                theme="urban",
                scenes=[original_scene]
            )
            
            pm.save_storyboard(project_name, storyboard)
            
            # Update scene with partial data
            pm.update_scene_in_storyboard(project_name, 1, updates)
            
            # Load updated storyboard
            updated_storyboard = pm.load_storyboard(project_name)
            updated_scene = updated_storyboard.scenes[0]
            
            # Verify updated fields changed
            for field, value in updates.items():
                assert getattr(updated_scene, field) == value
            
            # Verify non-updated fields preserved
            preserved_fields = {
                "id": 1,
                "start_time": 0.0,
                "end_time": 5.0,
                "duration": 5.0,
                "character_action": "original action",
                "video_prompt": "Original prompt"
            }
            
            for field, expected_value in preserved_fields.items():
                if field not in updates:
                    assert getattr(updated_scene, field) == expected_value
                    
        finally:
            shutil.rmtree(temp_dir)
    
    @given(
        st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            min_size=1,
            max_size=3
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_style_variation_preservation(
        self, style_variations
    ):
        """Property 25: Style Variation Preservation - Different styles are maintained across scenes."""
        import uuid
        import tempfile
        import shutil
        
        # Create isolated temp directory for this test
        temp_dir = tempfile.mkdtemp()
        try:
            pm = ProjectManager(temp_dir)
            
            # Create project with unique name
            audio_path = Path(temp_dir) / "sample.mp3"
            audio_path.write_bytes(b"fake audio data")
            project_name = f"Test_Project_{uuid.uuid4().hex[:8]}"
            pm.create_project(str(audio_path), project_name)
            
            # Create scenes with different styles
            scenes = []
            for i, style in enumerate(style_variations):
                character = CharacterDescriptor(
                    appearance=f"character {i}",
                    clothing=f"clothing {i}",
                    style=style
                )
                
                scene = SceneDescription(
                    id=i + 1,
                    start_time=i * 5.0,
                    end_time=(i + 1) * 5.0,
                    duration=5.0,
                    description=f"Scene {i}",
                    character_action="action",
                    camera_angle="medium",
                    lighting="natural",
                    environment="environment",
                    video_prompt=f"Prompt {i}",
                    character_descriptor=character
                )
                scenes.append(scene)
            
            storyboard = Storyboard(
                project_name=project_name,
                audio_file="test.mp3",
                audio_duration=len(style_variations) * 5.0,
                global_style="mixed",
                theme="varied",
                scenes=scenes
            )
            
            # Save and reload storyboard
            pm.save_storyboard(project_name, storyboard)
            loaded_storyboard = pm.load_storyboard(project_name)
            
            # Verify style variations are preserved
            for i, expected_style in enumerate(style_variations):
                scene = loaded_storyboard.scenes[i]
                assert scene.character_descriptor.style == expected_style
                
        finally:
            shutil.rmtree(temp_dir)
    
    @given(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=2,
            max_size=5
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_selective_scene_regeneration(
        self, scene_ids
    ):
        """Property 26: Selective Scene Regeneration - Individual scenes can be updated without affecting others."""
        import uuid
        import tempfile
        import shutil
        
        # Create isolated temp directory for this test
        temp_dir = tempfile.mkdtemp()
        try:
            pm = ProjectManager(temp_dir)
            
            # Create project with unique name
            audio_path = Path(temp_dir) / "sample.mp3"
            audio_path.write_bytes(b"fake audio data")
            project_name = f"Test_Project_{uuid.uuid4().hex[:8]}"
            pm.create_project(str(audio_path), project_name)
            
            # Create storyboard with multiple scenes
            scenes = []
            for scene_id in scene_ids:
                scene = SceneDescription(
                    id=scene_id,
                    start_time=(scene_id - 1) * 5.0,
                    end_time=scene_id * 5.0,
                    duration=5.0,
                    description=f"Original scene {scene_id}",
                    character_action="original action",
                    camera_angle="medium",
                    lighting="natural",
                    environment="environment",
                    video_prompt=f"Original prompt {scene_id}"
                )
                scenes.append(scene)
            
            storyboard = Storyboard(
                project_name=project_name,
                audio_file="test.mp3",
                audio_duration=len(scene_ids) * 5.0,
                global_style="anime",
                theme="urban",
                scenes=scenes
            )
            
            pm.save_storyboard(project_name, storyboard)
            
            # Update one scene
            target_scene_id = scene_ids[0]
            updates = {
                "description": f"Updated scene {target_scene_id}",
                "video_prompt": f"Updated prompt {target_scene_id}"
            }
            
            pm.update_scene_in_storyboard(project_name, target_scene_id, updates)
            
            # Load updated storyboard
            updated_storyboard = pm.load_storyboard(project_name)
            
            # Verify target scene was updated
            target_scene = next(s for s in updated_storyboard.scenes if s.id == target_scene_id)
            assert target_scene.description == f"Updated scene {target_scene_id}"
            assert target_scene.video_prompt == f"Updated prompt {target_scene_id}"
            
            # Verify other scenes unchanged
            for scene in updated_storyboard.scenes:
                if scene.id != target_scene_id:
                    assert scene.description == f"Original scene {scene.id}"
                    assert scene.video_prompt == f"Original prompt {scene.id}"
                    
        finally:
            shutil.rmtree(temp_dir)
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_project_listing_completeness_27(
        self, num_projects
    ):
        """Property 27: Project Listing Completeness - All projects appear in listing with metadata."""
        import uuid
        import tempfile
        import shutil
        
        # Create isolated temp directory for this test
        temp_dir = tempfile.mkdtemp()
        try:
            pm = ProjectManager(temp_dir)
            
            # Create sample audio file
            audio_path = Path(temp_dir) / "sample.mp3"
            audio_path.write_bytes(b"fake audio data")
            
            # Create multiple projects with unique names
            created_projects = []
            for i in range(num_projects):
                project_name = f"Project_{i}_{uuid.uuid4().hex[:8]}"
                pm.create_project(str(audio_path), project_name)
                created_projects.append(project_name)
            
            # List projects
            projects = pm.list_projects()
            
            # All created projects should be listed
            listed_names = [p["name"] for p in projects]
            for project_name in created_projects:
                assert project_name in listed_names
            
            # Each project should have required metadata
            for project in projects:
                assert "name" in project
                assert "created_at" in project
                assert "project_dir" in project
                assert "size_mb" in project
                assert "status" in project
                
                # Verify project directory exists
                assert Path(project["project_dir"]).exists()
                
        finally:
            shutil.rmtree(temp_dir)
    
    @given(st.text(min_size=1, max_size=30).filter(lambda x: x.strip() and x.isascii() and all(c not in '<>:"/\\|?*' for c in x)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_project_deletion_cleanup_28(
        self, project_name
    ):
        """Property 28: Project Deletion Cleanup - All project files are removed on deletion."""
        import uuid
        import tempfile
        import shutil
        
        # Create isolated temp directory for this test
        temp_dir = tempfile.mkdtemp()
        try:
            pm = ProjectManager(temp_dir)
            
            # Create sample audio file
            audio_path = Path(temp_dir) / "sample.mp3"
            audio_path.write_bytes(b"fake audio data")
            
            # Make project name unique
            unique_project_name = f"{project_name}_{uuid.uuid4().hex[:8]}"
            
            # Create project
            result = pm.create_project(str(audio_path), unique_project_name)
            project_dir = Path(result["project_dir"])
            
            # Add various files to project
            storyboard_file = project_dir / "storyboard.json"
            storyboard_file.write_text('{"test": "data"}')
            
            clips_dir = project_dir / "clips"
            clip_file = clips_dir / "scene_001.mp4"
            clip_file.write_bytes(b"fake clip data")
            
            final_dir = project_dir / "final"
            final_file = final_dir / "final.mp4"
            final_file.write_bytes(b"fake final video")
            
            temp_dir_proj = project_dir / "temp"
            temp_file = temp_dir_proj / "temp.txt"
            temp_file.write_text("temp data")
            
            # Verify all files exist
            assert storyboard_file.exists()
            assert clip_file.exists()
            assert final_file.exists()
            assert temp_file.exists()
            assert project_dir.exists()
            
            # Delete project
            pm.delete_project(unique_project_name)
            
            # Verify complete cleanup - no files should remain
            assert not storyboard_file.exists()
            assert not clip_file.exists()
            assert not final_file.exists()
            assert not temp_file.exists()
            assert not project_dir.exists()
            
            # Verify project not in listing
            projects = pm.list_projects()
            listed_names = [p["name"] for p in projects]
            assert unique_project_name not in listed_names
            
        finally:
            shutil.rmtree(temp_dir)


# Stateful testing
class ProjectManagerStateMachine(RuleBasedStateMachine):
    """Stateful testing for ProjectManager."""
    
    def __init__(self):
        super().__init__()
        self.temp_dir = tempfile.mkdtemp()
        self.project_manager = ProjectManager(self.temp_dir)
        self.created_projects = set()
        self.audio_path = Path(self.temp_dir) / "sample.mp3"
        self.audio_path.write_bytes(b"fake audio data")
    
    @rule(project_name=st.text(min_size=1, max_size=30).filter(lambda x: x.strip() and x.isascii() and all(c not in '<>:"/\\|?*' for c in x)))
    def create_project(self, project_name):
        """Create a new project."""
        assume(project_name not in self.created_projects)
        
        try:
            result = self.project_manager.create_project(str(self.audio_path), project_name)
            self.created_projects.add(project_name)
            
            # Verify project was created
            assert Path(result["project_dir"]).exists()
            
        except ProjectError:
            # Project creation can fail for various reasons
            pass
    
    @rule()
    def delete_random_project(self):
        """Delete a random existing project."""
        if self.created_projects:
            project_name = list(self.created_projects)[0]  # Take first project
            try:
                self.project_manager.delete_project(project_name)
                self.created_projects.remove(project_name)
            except ProjectError:
                # Project might have been deleted already
                pass
    
    @rule()
    def list_projects(self):
        """List all projects."""
        projects = self.project_manager.list_projects()
        listed_names = {p["name"] for p in projects}
        
        # All created projects should be in the list
        assert self.created_projects.issubset(listed_names)
    
    @invariant()
    def projects_directory_exists(self):
        """Project directory should always exist."""
        assert self.project_manager.projects_root.exists()
    
    def teardown(self):
        """Clean up after testing."""
        shutil.rmtree(self.temp_dir)


# Integration tests
class TestProjectManagerIntegration:
    """Integration tests for ProjectManager with other components."""
    
    @pytest.fixture
    def temp_projects_dir(self):
        """Create temporary projects directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def project_manager(self, temp_projects_dir):
        """Create ProjectManager instance."""
        return ProjectManager(temp_projects_dir)
    
    def test_full_project_workflow(self, project_manager, temp_projects_dir):
        """Test complete project workflow."""
        # Create audio file
        audio_path = Path(temp_projects_dir) / "sample.mp3"
        audio_path.write_bytes(b"fake audio data")
        
        # 1. Create project
        result = project_manager.create_project(str(audio_path), "Full Workflow Test")
        assert result["project_info"]["status"] == ProcessingStep.ANALYSIS.value
        
        # 2. Create and save storyboard
        character = CharacterDescriptor(
            appearance="young woman",
            clothing="red jacket",
            style="anime"
        )
        
        scenes = []
        for i in range(3):
            scene = SceneDescription(
                id=i + 1,
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                duration=10.0,
                description=f"Scene {i + 1}",
                character_action="dancing",
                camera_angle="medium shot",
                lighting="natural",
                environment="stage",
                video_prompt=f"Scene {i + 1} prompt",
                character_descriptor=character
            )
            scenes.append(scene)
        
        storyboard = Storyboard(
            project_name="Full Workflow Test",
            audio_file=str(audio_path),
            audio_duration=30.0,
            global_style="anime",
            theme="music video",
            scenes=scenes
        )
        
        project_manager.save_storyboard("Full Workflow Test", storyboard)
        
        # Check status after storyboard
        status = project_manager.get_project_status("Full Workflow Test")
        assert status.current_step == ProcessingStep.PLANNING.value
        assert status.total_scenes == 3
        
        # 3. Save generated clips
        for i in range(3):
            clip_path = Path(temp_projects_dir) / f"clip_{i}.mp4"
            clip_path.write_bytes(b"fake video data")
            
            project_manager.save_generated_clip("Full Workflow Test", i + 1, str(clip_path))
        
        # Check status after clips
        status = project_manager.get_project_status("Full Workflow Test")
        assert status.current_step == ProcessingStep.GENERATION.value
        assert status.scenes_generated == 3
        
        # 4. Save final video
        final_video_path = Path(temp_projects_dir) / "final.mp4"
        final_video_path.write_bytes(b"final video data")
        
        project_manager.save_final_video("Full Workflow Test", str(final_video_path))
        
        # Check final status
        status = project_manager.get_project_status("Full Workflow Test")
        assert status.current_step == ProcessingStep.COMPLETE.value
        assert status.progress_percent == 100.0
        
        # 5. Cleanup
        project_manager.cleanup_project("Full Workflow Test")
        
        # 6. Verify project still exists and is complete
        projects = project_manager.list_projects()
        assert len(projects) == 1
        assert projects[0]["name"] == "Full Workflow Test"
        assert projects[0]["status"] == ProcessingStep.COMPLETE.value


# Run stateful tests
TestProjectManagerStateful = ProjectManagerStateMachine.TestCase
import argparse
import os
import sys
from director import Director
from artist import Artist
from editor import Editor

def main():
    parser = argparse.ArgumentParser(description="VibeFrame: AI Music Video Generator")
    parser.add_argument("audio_file", help="Path to input audio file")
    parser.add_argument("--project_dir", default="project_output", help="Directory to save project files")
    parser.add_argument("--interval", type=float, default=5.0, help="Minimum seconds between cuts")
    parser.add_argument("--skip-enhance", action="store_true", help="Skip AI prompt enhancement")
    parser.add_argument("--auto", action="store_true", help="Run full pipeline without pausing for manual storyboard edit")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file {args.audio_file} not found.")
        sys.exit(1)
        
    os.makedirs(args.project_dir, exist_ok=True)
    
    # 1. The Director
    print("\nXXX SCENE 1: THE DIRECTOR XXX")
    director = Director(args.audio_file)
    director.load_audio()
    cuts = director.detect_beats(min_interval=args.interval)
    
    storyboard_path = os.path.join(args.project_dir, "storyboard.json")
    director.generate_storyboard(cuts, storyboard_path)
    
    # Pause for user editing
    if not args.auto:
        print(f"\n[PAUSE] Storyboard generated at: {storyboard_path}")
        print("You can now edit the 'description' and 'visual_style' fields in the JSON file.")
        print("Add specific instructions for each scene.")
        input("Press Enter to continue to The Artist...")
        
    # 2. The Artist
    print("\nXXX SCENE 2: THE ARTIST XXX")
    artist = Artist()
    
    if not args.skip_enhance:
        artist.enhance_prompts(storyboard_path)
        
    images_dir = os.path.join(args.project_dir, "frames")
    artist.generate_images(storyboard_path, images_dir)
    
    # 3. The Editor
    print("\nXXX SCENE 3: THE EDITOR XXX")
    editor = Editor()
    output_video = os.path.join(args.project_dir, "final_video.mp4")
    editor.create_video(storyboard_path, output_video)
    
    print(f"\n[SUCCESS] Music Video generated: {output_video}")

if __name__ == "__main__":
    main()

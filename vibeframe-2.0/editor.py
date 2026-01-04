import json
import os
import argparse
from moviepy.editor import *
from typing import List, Tuple

class Editor:
    def __init__(self):
        pass

    def create_video(self, storyboard_path: str, output_file: str = "output_video.mp4"):
        print(f"Reading storyboard from {storyboard_path}...")
        with open(storyboard_path, 'r') as f:
            data = json.load(f)

        audio_path = data.get("audio_file")
        scenes = data.get("scenes", [])
        
        if not os.path.exists(audio_path):
             print(f"Error: Audio file not found at {audio_path}")
             return

        print("Loading audio...")
        audio_clip = AudioFileClip(audio_path)
        video_clips = []

        print("Assembling scenes...")
        for scene in scenes:
            image_path = scene.get("image_path")
            duration = scene.get("duration")
            
            if not image_path or not os.path.exists(image_path):
                print(f"Warning: Image for Scene {scene['id']} missing ({image_path}). Skipping or using black frame.")
                # TODO: Insert black frame or placeholder
                color_clip = ColorClip(size=(1920, 1080), color=(0,0,0), duration=duration)
                video_clips.append(color_clip)
                continue
                
            # Create Image Clip
            img_clip = ImageClip(image_path).set_duration(duration)
            
            # Resize to fit 1080p (maintain aspect ratio, black bars if needed, or crop)
            # Simple approach: resize height to 1080, then crop/resize width
            img_clip = img_clip.resize(height=1080)
            if img_clip.w < 1920:
                 img_clip = img_clip.resize(width=1920)
            img_clip = img_clip.crop(x1=img_clip.w/2 - 1920/2, width=1920, height=1080)

            # Apply subtle zoom (Ken Burns)
            # Note: MoviePy zoom can be tricky. Using a simple static clip for now if speed is priority.
            # To add zoom: would need a custom transformation function.
            # Let's add a very simple "zoom in" effect by resizing over time.
            
            clip_w, clip_h = img_clip.size
            def zoom(get_frame, t):
                scale = 1 + 0.04 * (t / duration) # 4% zoom over the duration
                frame = get_frame(t)
                # This is complex in raw moviepy without custom resize per frame which is slow.
                # Sticking to static image for robust MVP.
                return frame
            
            # Alternative: Pan effect? 
            # For MVP speed, let's keep it static but high quality.
            
            video_clips.append(img_clip)

        print("Concatenating video clips...")
        # Concatenate
        final_video = concatenate_videoclips(video_clips, method="compose")
        
        # Set Audio (trim if video is shorter/longer)
        if final_video.duration > audio_clip.duration:
             final_video = final_video.subclip(0, audio_clip.duration)
        else:
             audio_clip = audio_clip.subclip(0, final_video.duration)
             
        final_video = final_video.set_audio(audio_clip)

        print(f"Writing video file to {output_file}...")
        final_video.write_videofile(output_file, fps=24, codec="libx264", audio_codec="aac")
        print("Video creation complete!")

def main():
    parser = argparse.ArgumentParser(description="The Editor: Video Compilation")
    parser.add_argument("storyboard", help="Path to storyboard.json")
    parser.add_argument("--output", default="final_music_video.mp4", help="Output video path")
    
    args = parser.parse_args()
    
    editor = Editor()
    editor.create_video(args.storyboard, args.output)

if __name__ == "__main__":
    main()

import librosa
import numpy as np
import json
import os
import argparse
from typing import List, Dict

class Director:
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.y = None
        self.sr = None
        self.duration = 0

    def load_audio(self):
        """Loads the audio file."""
        print(f"Loading audio from {self.audio_path}...")
        self.y, self.sr = librosa.load(self.audio_path)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        print(f"Audio loaded. Duration: {self.duration:.2f} seconds.")

    def detect_beats(self, min_interval: float = 4.0) -> List[float]:
        """
        Detects significant beat drops or cuts.
        
        Args:
            min_interval: Minimum time (in seconds) between cuts to avoid frenetic editing.
        
        Returns:
            List of timestamps (floats) where cuts should happen.
        """
        print("Analyzing beats and rhythm...")
        tempo, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        
        # Simple algorithm: Pick a beat every ~min_interval seconds
        # A more advanced version would look for `onset_strength` peaks.
        
        cut_times = [0.0]
        last_cut = 0.0
        
        for t in beat_times:
            if t - last_cut >= min_interval:
                cut_times.append(float(t))
                last_cut = t
                
        # Ensure the last scene goes to the end
        if self.duration - cut_times[-1] > 1.0: # if gap is significant
             pass # last scene handles itself until end
             
        return cut_times

    def generate_storyboard(self, cut_times: List[float], output_path: str = "storyboard.json"):
        """Creates the JSON structure for the user to fill in."""
        scenes = []
        
        for i, start_time in enumerate(cut_times):
            end_time = cut_times[i+1] if i+1 < len(cut_times) else self.duration
            duration = end_time - start_time
            
            scene = {
                "id": i + 1,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "duration": round(duration, 2),
                "description": "",  # To be filled by user or AI
                "visual_style": "Consistent character, cinematic lighting", # Default style
                "image_path": ""    # Placeholder for generated image
            }
            scenes.append(scene)
            
        storyboard = {
            "project_name": "My AI Music Video",
            "audio_file": self.audio_path,
            "fps": 24,
            "scenes": scenes
        }
        
        with open(output_path, 'w') as f:
            json.dump(storyboard, f, indent=4)
            
        print(f"Storyboard generated with {len(scenes)} scenes at: {output_path}")
        return storyboard

def main():
    parser = argparse.ArgumentParser(description="The Director: Audio Analysis for Music Video Generation")
    parser.add_argument("audio_input", help="Path to input audio file")
    parser.add_argument("--interval", type=float, default=5.0, help="Minimum seconds between cuts")
    parser.add_argument("--output", default="storyboard.json", help="Output JSON path")
    
    args = parser.parse_args()
    
    director = Director(args.audio_input)
    director.load_audio()
    cuts = director.detect_beats(min_interval=args.interval)
    director.generate_storyboard(cuts, args.output)

if __name__ == "__main__":
    main()

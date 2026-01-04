import json
import os
import requests
import argparse
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv()

class Artist:
    def __init__(self):
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        # Using a reliable free model on HF Inference API (SDXL Turbo is fast and often free)
        self.hf_model_url = "https://api-inference.huggingface.co/models/stabilityai/sdxl-turbo"

    def enhance_prompts(self, storyboard_path: str):
        """
        Reads storyboard, checks for missing 'image_prompt', and uses AI to generate them
        based on the 'description' and 'visual_style'.
        """
        with open(storyboard_path, 'r') as f:
            data = json.load(f)
            
        scenes = data.get("scenes", [])
        project_style = data.get("scenes", [{}])[0].get("visual_style", "Cinematic")
        
        print(f"Enhancing prompts for {len(scenes)} scenes using OpenRouter...")
        
        for scene in scenes:
            if not scene.get("image_prompt"): # Only generate if missing
                description = scene.get("description", "Abstract visuals matching the beat")
                style = scene.get("visual_style", project_style)
                
                print(f"Generating prompt for Scene {scene['id']}...")
                
                # Call LLM to create a stable diffusion prompt
                prompt = self._call_llm_for_prompt(description, style)
                scene["image_prompt"] = prompt
                
        # Save back
        with open(storyboard_path, 'w') as f:
            json.dump(data, f, indent=4)
        print("Prompts enhanced and saved.")

    def _call_llm_for_prompt(self, description: str, style: str) -> str:
        """Helper to call OpenRouter."""
        if not self.openrouter_key:
            print("Warning: No OpenRouter Key. Using raw description.")
            return f"{style}, {description}"
            
        payload = {
            "model": "google/gemini-2.0-flash-exp:free",
            "messages": [{
                "role": "user",
                "content": f"Convert this scene description into a high-quality Stable Diffusion image prompt. Keep it under 40 words. Focus on visual details, lighting, and style. \n\nStyle: {style}\nDescription: {description}\n\nPrompt:"
            }]
        }
        
        try:
            response = requests.post(
                self.openrouter_url,
                headers={"Authorization": f"Bearer {self.openrouter_key}"},
                json=payload
            )
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return content.strip()
            else:
                print(f"LLM Error: {response.status_code}")
                return f"{style}, {description}"
        except Exception as e:
            print(f"LLM Exception: {e}")
            return f"{style}, {description}"

    def generate_images(self, storyboard_path: str, output_dir: str = "generated_images"):
        """
        Iterates through storyboard and generates images for each scene using HF API.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(storyboard_path, 'r') as f:
            data = json.load(f)
            
        scenes = data.get("scenes", [])
        print(f"Starting image generation for {len(scenes)} scenes...")
        
        for scene in scenes:
            prompt = scene.get("image_prompt")
            scene_id = scene.get("id")
            output_filename = os.path.join(output_dir, f"scene_{scene_id:03d}.png")
            
            # Skip if already exists
            if os.path.exists(output_filename):
                print(f"Scene {scene_id} image exists. Skipping.")
                scene["image_path"] = output_filename
                continue
                
            print(f"Generating image for Scene {scene_id}: {prompt[:30]}...")
            success = self._call_hf_inference(prompt, output_filename)
            
            if success:
                scene["image_path"] = output_filename
            else:
                print(f"Failed to generate for Scene {scene_id}")

        # Save paths back to storyboard
        with open(storyboard_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    def _call_hf_inference(self, prompt: str, output_path: str) -> bool:
        """Calls HuggingFace Inference API."""
        if not self.hf_token:
            print("Error: No HuggingFace Token provided.")
            return False
            
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {"inputs": prompt}
        
        try:
            response = requests.post(self.hf_model_url, headers=headers, json=payload)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return True
            else:
                print(f"HF Error {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"HF Exception: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="The Artist: Image Generation")
    parser.add_argument("storyboard", help="Path to storyboard.json")
    parser.add_argument("--enhance", action="store_true", help="Enhance prompts using LLM first")
    parser.add_argument("--generate", action="store_true", help="Generate images from prompts")
    parser.add_argument("--output_dir", default="frames", help="Output directory for images")
    
    args = parser.parse_args()
    
    artist = Artist()
    
    if args.enhance:
        artist.enhance_prompts(args.storyboard)
        
    if args.generate:
        artist.generate_images(args.storyboard, args.output_dir)

if __name__ == "__main__":
    main()

"""
Create a simple test video WITHOUT text (no ImageMagick needed)
"""
from moviepy.editor import ColorClip
import numpy as np

print("Creating test video...")

# Create a 15-second colorful video (no text needed)
duration = 15
clip = ColorClip(size=(1920, 1080), color=(70, 130, 220), duration=duration)

# Export
output_path = "uploads/test_video.mp4"
clip.write_videofile(output_path, fps=24, codec='libx264', audio=False)

print(f"✓ Test video created: {output_path}")
print(f"  Duration: {duration}s")
print(f"  Resolution: 1920x1080")
print("\nNow test with:")
print('  python test_cli.py uploads/test_video.mp4 "Make vibrant for Instagram"')
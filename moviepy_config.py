"""
MoviePy ImageMagick Configuration
"""
import os

# Set ImageMagick path for MoviePy
IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"

# Configure MoviePy
os.environ['IMAGEMAGICK_BINARY'] = IMAGEMAGICK_BINARY

print(f"✓ ImageMagick configured: {IMAGEMAGICK_BINARY}")

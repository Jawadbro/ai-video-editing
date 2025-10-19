"""
═══════════════════════════════════════════════════════════════════════════════
AGENTIC VIDEO EDITING AI - POWERED BY GOOGLE GEMINI
═══════════════════════════════════════════════════════════════════════════════

This agent uses Few-Shot Learning to understand ANY editing style through
natural language. It learns from professional examples and generalizes to
new requests.

Author: Your Name
Version: 2.0 - True Agentic Intelligence
License: MIT
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
from typing import Dict, Any, Optional
from models import ProcessingOptions
from dotenv import load_dotenv

# If you want to use Gemini API, import and configure here
import google.generativeai as genai

load_dotenv()

class VideoEditingAgent:
    """AI Agent for video editing"""

    def __init__(self):
        self.prompt = self._build_learning_prompt()

    def _build_learning_prompt(self) -> str:
        """
        Builds the system prompt with examples and instructions
        """
        return """
You are an AI video editing assistant. Your job is to analyze user instructions and create a detailed editing plan.

Each editing plan should be a JSON object with the following keys:
- lut: string (optional) - Select a LUT for overall color grading
- brightness: float (0.8-1.5, default 1.2)
- contrast: float (0.9-1.5, default 1.1)
- saturation: float (0.8-1.8, default 1.3)
- color_tone: "warm", "cool", "neutral", or null
- captions: boolean (default true)
- formats: list of strings (e.g., ["youtube", "instagram_story", "tiktok"])
- special_effects: list of strings (optional - e.g., ["vignette", "grain"])
- text_overlay: string or null (optional)
- text_position: "top", "bottom", "center" (optional)
- min_scene_duration: float (default 2.0)
- transition_duration: float (default 0.5)

## LUT Support:
Use LUTs (Look-Up Tables) for specific cinematic looks:
- 'cinematic.cube': Teal-orange Hollywood blockbuster style
- 'cool_teal.cube': Modern, professional, clean style
- 'warm_orange.cube': Cozy, warm tones
- 'vogue_editorial.cube': High-fashion, desaturated cool tones

## Examples:

User: "Make it cinematic"
Output: {"lut": "cinematic.cube", "brightness": 1.1, "contrast": 1.3, "saturation": 1.4, "formats": ["youtube"]}

User: "I want a high-fashion Vogue magazine style"
Output: {"lut": "vogue_editorial.cube", "formats": ["instagram_feed"], "special_effects": ["vignette"]}

User: "Vibrant and cool for TikTok"
Output: {"lut": "cool_teal.cube", "brightness": 1.3, "contrast": 1.2, "saturation": 1.5, "formats": ["tiktok"]}

User: "Add captions and a call-to-action at the bottom"
Output: {"captions": true, "text_overlay": "Subscribe now!", "text_position": "bottom"}

User: "Just basic color correction"
Output: {"brightness": 1.2, "contrast": 1.1, "saturation": 1.3, "formats": ["youtube", "instagram_story"]}
"""

    def plan_workflow(self, instruction: str, video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interprets user instruction and generates a JSON editing plan.
        Replace this logic with a call to Gemini or your LLM for production.
        """
        # Example: Simulated logic for demonstration
        instr = instruction.lower()
        if "cinematic" in instr:
            return {
                "lut": "cinematic.cube",
                "brightness": 1.1,
                "contrast": 1.3,
                "saturation": 1.4,
                "formats": ["youtube"]
            }
        elif "vogue" in instr:
            return {
                "lut": "vogue_editorial.cube",
                "formats": ["instagram_feed"],
                "special_effects": ["vignette"]
            }
        elif "tiktok" in instr and "vibrant" in instr:
            return {
                "lut": "cool_teal.cube",
                "brightness": 1.3,
                "contrast": 1.2,
                "saturation": 1.5,
                "formats": ["tiktok"]
            }
        elif "caption" in instr or "call-to-action" in instr:
            return {
                "captions": True,
                "text_overlay": "Subscribe now!",
                "text_position": "bottom"
            }
        else:
            # Default fallback
            return {
                "brightness": 1.2,
                "contrast": 1.1,
                "saturation": 1.3,
                "formats": ["youtube", "tiktok"]
            }

    def plan_to_options(self, plan: Dict[str, Any]) -> ProcessingOptions:
        """
        Convert the agent's plan (dict) to a ProcessingOptions object.
        """
        return ProcessingOptions(
            lut=plan.get('lut'),
            brightness=float(plan.get('brightness', 1.2)),
            contrast=float(plan.get('contrast', 1.1)),
            saturation=float(plan.get('saturation', 1.3)),
            color_tone=plan.get('color_tone', None),
            captions=plan.get('captions', True),
            formats=plan.get('formats', ["youtube"]),
            special_effects=plan.get('special_effects', []),
            text_overlay=plan.get('text_overlay'),
            text_position=plan.get('text_position', 'bottom'),
            min_scene_duration=float(plan.get('min_scene_duration', 2.0)),
            transition_duration=float(plan.get('transition_duration', 0.5))
        )

    def display_reasoning(self, plan: Dict[str, Any]):
        """
        Print the agent's reasoning and plan for debugging and transparency.
        """
        print("\n📋 Editing Plan:")
        if plan.get('lut'):
            print(f"   ✓ LUT Applied: {plan['lut']}")
        else:
            print(f"   ✓ Color Grading: brightness={plan.get('brightness', 1.2)}, contrast={plan.get('contrast', 1.1)}, saturation={plan.get('saturation', 1.3)}")
        if plan.get('color_tone'):
            print(f"   ✓ Color Tone: {plan['color_tone']}")
        if plan.get('captions'):
            print("   ✓ Captions: Enabled")
        if plan.get('text_overlay'):
            print(f"   ✓ Text Overlay: '{plan['text_overlay']}' at {plan.get('text_position', 'bottom')}")
        if plan.get('special_effects'):
            print(f"   ✓ Special Effects: {plan['special_effects']}")
        print(f"   ✓ Export Formats: {plan.get('formats', ['youtube'])}")
        print(f"   ✓ Scene Duration: {plan.get('min_scene_duration', 2.0)}s minimum")
        print(f"   ✓ Transitions: {plan.get('transition_duration', 0.5)}s")

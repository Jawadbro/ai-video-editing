"""
LangGraph Nodes - Individual processing steps
Production ready with proper state management
"""

import moviepy_config
from typing import Dict
from graph_state import VideoEditingState
from video_processor import VideoProcessor
from utils import QualityAssurance
from agent import VideoEditingAgent
from models import ProcessingOptions, SceneInfo, SilenceInfo
import os

# Initialize tools (shared across all nodes)
processor = VideoProcessor()
qa = QualityAssurance()


# ========== NODE 1: AGENT PLANNING ==========

def agent_planning_node(state: VideoEditingState) -> Dict:
    """
    Gemini agent analyzes request and plans workflow
    """
    print("\n[NODE 1/9] 🤖 Agent Planning...")
    
    try:
        agent = VideoEditingAgent()
        
        # Extract metadata first for context
        metadata = processor.extract_metadata(state['input_path'])
        
        # Agent creates plan
        options = agent.plan_workflow(
            state['user_instruction'],
            video_metadata={
                'duration': float(metadata.duration),
                'resolution': metadata.resolution,
                'has_audio': bool(metadata.has_audio),
                'fps': float(metadata.fps)
            }
        )
        
        # Convert ProcessingOptions (now a dict) to regular dict for state
        if isinstance(options, dict):
            editing_options = dict(options)
        else:
            editing_options = options.__dict__
        
        # Log unknown keys for future improvements
        known_keys = {
            'color_grading', 'brightness', 'contrast', 'saturation', 
            'captions', 'burn_captions', 'caption_fontsize', 
            'text_overlay', 'text_position', 'text_fontsize', 
            'formats', 'min_scene_duration', 'transition_duration', 
            'detect_black_frames', 'detect_audio_clipping'
        }
        unknown_keys = set(editing_options.keys()) - known_keys
        if unknown_keys:
            print(f"   ℹ️ Agent suggested new options: {unknown_keys} (will be logged)")
        
        return {
            'metadata': {
                'duration': float(metadata.duration),
                'resolution': metadata.resolution,
                'fps': float(metadata.fps),
                'has_audio': bool(metadata.has_audio),
                'file_size': int(metadata.file_size)
            },
            'editing_options': editing_options,
            'current_stage': 'planned',
            'progress': 10
        }
    
    except Exception as e:
        return {
            'errors': [f"Agent planning failed: {str(e)}"],
            'current_stage': 'failed',
            'progress': 0
        }


# ========== NODE 2: VALIDATION ==========

def validate_input_node(state: VideoEditingState) -> Dict:
    """
    Validate video file integrity
    """
    print("\n[NODE 2/9] ✓ Validating Input...")
    
    is_valid, message = qa.validate_video_file(state['input_path'])
    
    if not is_valid:
        print(f"   ✗ Validation failed: {message}")
        return {
            'errors': [f"Validation failed: {message}"],
            'current_stage': 'failed'
        }
    
    print(f"   ✓ {message}")
    return {
        'current_stage': 'validated',
        'progress': 15
    }


# ========== NODE 3: ANALYSIS ==========

def analyze_video_node(state: VideoEditingState) -> Dict:
    """
    Analyze scenes, silences, and generate thumbnail
    """
    print("\n[NODE 3/9] 📊 Analyzing Video...")
    
    try:
        # Detect scenes
        scenes = processor.detect_scenes(state['input_path'])
        print(f"   Found {len(scenes)} scenes")
        
        # Detect silences
        silences = processor.detect_silences(state['input_path'])
        print(f"   Found {len(silences)} silent sections")
        
        # Generate thumbnail
        thumb_path = f"temp/{state['video_id']}_thumb.jpg"
        processor.generate_thumbnail(state['input_path'], thumb_path)
        print(f"   Thumbnail: {thumb_path}")
        
        # Convert to dicts for JSON serialization (convert numpy types to native Python)
        scenes_dict = [
            {
                'start': float(s.start),
                'end': float(s.end),
                'duration': float(s.duration),
                'kept': bool(s.kept)
            }
            for s in scenes
        ]
        
        silences_dict = [
            {
                'start': float(s.start),
                'end': float(s.end),
                'duration': float(s.duration)
            }
            for s in silences
        ]
        
        return {
            'scenes': scenes_dict,
            'silences': silences_dict,
            'thumbnail_path': thumb_path,
            'current_stage': 'analyzed',
            'progress': 30
        }
    
    except Exception as e:
        return {
            'errors': [f"Analysis failed: {str(e)}"],
            'current_stage': 'failed'
        }


# ========== NODE 4: QUALITY CHECK ==========

def quality_check_node(state: VideoEditingState) -> Dict:
    """
    Check video quality and decide if enhancement needed
    """
    print("\n[NODE 4/9] 🔍 Quality Check...")
    
    issues = []
    score = 100.0
    
    # Check for black frames
    try:
        black_frames = qa.detect_black_frames(state['input_path'])
        if black_frames:
            issues.append(f"Found {len(black_frames)} black frame sequences")
            score -= 20
            print(f"   ⚠ {issues[-1]}")
    except Exception as e:
        print(f"   ⚠ Black frame detection skipped: {str(e)}")
    
    # Check scene count
    if len(state.get('scenes', [])) < 3:
        issues.append("Too few scenes detected")
        score -= 15
        print(f"   ⚠ {issues[-1]}")
    
    # Check duration
    metadata = state.get('metadata', {})
    if metadata.get('duration', 0) < 5:
        issues.append("Video too short")
        score -= 10
        print(f"   ⚠ {issues[-1]}")
    
    needs_enhancement = score < 70
    
    print(f"   Quality Score: {score}/100")
    if needs_enhancement:
        print(f"   ⚠ Video needs enhancement")
    else:
        print(f"   ✓ Quality acceptable")
    
    return {
        'quality_score': float(score),
        'quality_issues': issues,
        'needs_enhancement': bool(needs_enhancement),
        'current_stage': 'quality_checked',
        'progress': 40
    }


# ========== NODE 5: ENHANCEMENT ==========

def enhance_video_node(state: VideoEditingState) -> Dict:
    """
    Enhance video if quality is poor (conditional node)
    """
    print("\n[NODE 5/9] ✨ Enhancing Video...")
    
    # Boost color grading parameters
    options_dict = state['editing_options'].copy()
    options_dict['brightness'] = float(min(options_dict.get('brightness', 1.2) + 0.2, 1.5))
    options_dict['saturation'] = float(min(options_dict.get('saturation', 1.3) + 0.3, 1.8))
    options_dict['contrast'] = float(min(options_dict.get('contrast', 1.1) + 0.1, 1.4))
    
    print(f"   Boosted brightness to {options_dict['brightness']}")
    print(f"   Boosted saturation to {options_dict['saturation']}")
    
    return {
        'editing_options': options_dict,
        'current_stage': 'enhanced',
        'progress': 45
    }


# ========== NODE 6: EDITING (UPDATED - SAVES TO TEMP FILE) ==========

def edit_video_node(state: VideoEditingState) -> Dict:
    """
    Main editing: trim, transitions, color grade, text overlay
    SAVES TO TEMP FILE (not stored in state)
    """
    print("\n[NODE 6/9] ✂️ Editing Video...")
    
    try:
        options = ProcessingOptions(**state['editing_options'])
        
        # Convert scenes back to SceneInfo objects
        scenes = [SceneInfo(**s) for s in state['scenes']]
        silences = [SilenceInfo(**s) for s in state['silences']]
        
        # Auto-trim
        print("   Trimming scenes...")
        subclips, kept_scenes = processor.auto_trim(
            state['input_path'],
            scenes,
            silences,
            float(options.get('min_scene_duration', 2.0))
        )
        print(f"   Kept {len(kept_scenes)} scenes")
        
        # Apply transitions
        print("   Applying transitions...")
        edited_clip = processor.apply_transitions(subclips, float(options.get('transition_duration', 0.5)))
        
        # Color grading
        if options.get('color_grading', True):
            print("   Color grading...")
            edited_clip = processor.apply_color_grading(
                edited_clip,
                float(options.get('brightness', 1.2)),
                float(options.get('contrast', 1.1)),
                float(options.get('saturation', 1.3))
            )
        
        # Text overlay
        if options.get('text_overlay'):
            print(f"   Adding text: '{options.get('text_overlay')}'")
            edited_clip = processor.add_text_overlay(
                edited_clip,
                options.get('text_overlay'),
                options.get('text_position', 'bottom'),
                int(options.get('text_fontsize', 50))
            )
        
        # SAVE TO TEMP FILE (so state is serializable)
        temp_path = f"temp/{state['video_id']}_edited.mp4"
        print(f"   Saving to temp file...")
        edited_clip.write_videofile(
            temp_path,
            codec='libx264',
            audio_codec='aac',
            fps=24,
            logger=None
        )
        edited_clip.close()  # Close immediately to free memory
        print(f"   ✓ Saved: {temp_path}")
        
        # Convert kept scenes to dict
        kept_scenes_dict = [
            {
                'start': float(s.start),
                'end': float(s.end),
                'duration': float(s.duration),
                'kept': bool(s.kept)
            }
            for s in kept_scenes
        ]
        
        return {
            'temp_edited_path': temp_path,  # Store path, not object
            'kept_scenes': kept_scenes_dict,
            'current_stage': 'edited',
            'progress': 60
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'errors': [f"Editing failed: {str(e)}"],
            'current_stage': 'failed'
        }


# ========== NODE 7: CAPTIONS ==========

def generate_captions_node(state: VideoEditingState) -> Dict:
    """
    Generate captions with Whisper
    """
    print("\n[NODE 7/9] 💬 Generating Captions...")
    
    options = ProcessingOptions(**state['editing_options'])
    
    if not options.get('captions', True):
        print("   Captions disabled")
        return {
            'current_stage': 'captions_skipped',
            'progress': 70
        }
    
    try:
        srt_content, segments = processor.generate_captions(state['input_path'])
        
        srt_path = f"exports/{state['video_id']}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        print(f"   Generated {len(segments)} caption segments")
        print(f"   SRT file: {srt_path}")
        
        # Convert captions to dict
        captions_dict = [
            {
                'start': float(c.start),
                'end': float(c.end),
                'text': c.text
            }
            for c in segments
        ]
        
        return {
            'captions': captions_dict,
            'srt_path': srt_path,
            'current_stage': 'captions_generated',
            'progress': 75
        }
    
    except Exception as e:
        print(f"   ⚠ Caption generation failed: {str(e)}")
        # Don't fail the entire workflow, just skip captions
        return {
            'current_stage': 'captions_failed',
            'progress': 70
        }


# ========== NODE 8: EXPORT (UPDATED - LOADS FROM TEMP FILE) ==========

def export_node(state: VideoEditingState) -> Dict:
    """
    Export to multiple formats
    LOADS FROM TEMP FILE
    """
    print("\n[NODE 8/9] 📦 Exporting...")
    
    try:
        from moviepy.editor import VideoFileClip
        
        options = ProcessingOptions(**state['editing_options'])
        output_base = f"exports/{state['video_id']}"
        
        # Load from temp file
        print(f"   Loading edited video from: {state['temp_edited_path']}")
        edited_clip = VideoFileClip(state['temp_edited_path'])
        
        formats = options.get('formats', ['youtube', 'instagram_story'])
        print(f"   Exporting {len(formats)} format(s)...")
        exports = processor.export_multi_format(
            edited_clip,
            output_base,
            formats
        )
        
        # Close clip
        edited_clip.close()
        
        # Cleanup temp file
        if os.path.exists(state['temp_edited_path']):
            os.remove(state['temp_edited_path'])
            print(f"   Cleaned up temp file")
        
        # Convert to dicts
        exports_dict = [
            {
                'format_name': e.format_name,
                'filepath': e.filepath,
                'resolution': tuple(int(x) for x in e.resolution),
                'aspect_ratio': tuple(int(x) for x in e.aspect_ratio),
                'file_size': int(e.file_size)
            }
            for e in exports
        ]
        
        print(f"   ✓ Exported {len(exports)} videos")
        
        return {
            'exports': exports_dict,
            'current_stage': 'completed',
            'progress': 100
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'errors': [f"Export failed: {str(e)}"],
            'current_stage': 'failed'
        }


# ========== NODE 9: ERROR HANDLER ==========

def error_handler_node(state: VideoEditingState) -> Dict:
    """
    Handle errors and decide retry strategy
    """
    print("\n[NODE 9/9] ⚠️ Error Handler...")
    
    retry_count = state.get('retry_count', 0)
    max_retries = state.get('max_retries', 2)
    
    errors = state.get('errors', [])
    if errors:
        print(f"   Errors encountered:")
        for err in errors:
            print(f"     - {err}")
    
    if retry_count < max_retries:
        print(f"   Attempting retry {retry_count + 1}/{max_retries}")
        return {
            'retry_count': retry_count + 1,
            'current_stage': 'retrying'
        }
    else:
        print("   Max retries reached. Workflow failed.")
        return {
            'current_stage': 'failed'
        }

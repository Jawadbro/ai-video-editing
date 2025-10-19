import moviepy_config
import os
import cv2
import whisper
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, TextClip
from moviepy.video.fx import fadein, fadeout
from scenedetect import detect, ContentDetector
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from models import VideoMetadata, SceneInfo, SilenceInfo, CaptionSegment, ExportInfo

class VideoProcessor:
    """Enhanced video processor with detailed tracking"""
    
    def __init__(self):
        self.whisper_model = None
    
    # ========== METADATA EXTRACTION ==========
    
    def extract_metadata(self, video_path: str) -> VideoMetadata:
        """Extract comprehensive video metadata"""
        clip = VideoFileClip(video_path)
        file_size = os.path.getsize(video_path)
        
        metadata = VideoMetadata(
            duration=clip.duration,
            fps=clip.fps,
            width=clip.w,
            height=clip.h,
            resolution=f"{clip.w}x{clip.h}",
            has_audio=clip.audio is not None,
            file_size=file_size
        )
        
        clip.close()
        return metadata
    
    def generate_thumbnail(self, video_path: str, output_path: str, time: float = 1.0) -> str:
        """Generate thumbnail at specific time"""
        clip = VideoFileClip(video_path)
        
        # Ensure time is within video duration
        time = min(time, clip.duration - 0.1)
        
        frame = clip.get_frame(time)
        Image.fromarray(frame).save(output_path)
        clip.close()
        return output_path
    
    # ========== SCENE DETECTION ==========
    
    def detect_scenes(self, video_path: str, threshold: float = 27.0) -> List[SceneInfo]:
        """Detect scene changes with detailed info"""
        try:
            scenes = detect(video_path, ContentDetector(threshold=threshold))
            scene_list = []
            for start_time, end_time in scenes:
                start_sec = start_time.get_seconds()
                end_sec = end_time.get_seconds()
                scene_list.append(SceneInfo(
                    start=start_sec,
                    end=end_sec,
                    duration=end_sec - start_sec
                ))
            return scene_list
        except Exception as e:
            print(f"   ⚠️ Scene detection failed: {str(e)}")
            return []  # Return empty list on error
    
    def detect_silences(self, video_path: str, silence_thresh: float = -50, 
                   min_duration: float = 0.5) -> List[SilenceInfo]:
        """Detect silent parts in audio"""
        try:
            clip = VideoFileClip(video_path)
            if not clip.audio:
                clip.close()
                return []
            
            audio_array = clip.audio.to_soundarray(fps=22050)
            
            # Handle both mono and stereo audio
            if len(audio_array.shape) == 1:
                # Mono audio
                rms = np.abs(audio_array)
            else:
                # Stereo or multi-channel audio
                rms = np.sqrt(np.mean(audio_array**2, axis=1))
            
            threshold = 10**(silence_thresh/20)
            
            silences = []
            in_silence = False
            silence_start = 0
            
            for i, volume in enumerate(rms):
                time = i / 22050
                if volume < threshold:
                    if not in_silence:
                        silence_start = time
                        in_silence = True
                else:
                    if in_silence and (time - silence_start) > min_duration:
                        silences.append(SilenceInfo(
                            start=silence_start,
                            end=time,
                            duration=time - silence_start
                        ))
                    in_silence = False
            
            clip.close()
            return silences
        
        except Exception as e:
            print(f"   ⚠️ Silence detection failed: {str(e)}")
            return []  # Return empty list on error instead of crashing

    # ========== INTELLIGENT EDITING ==========
    
    def auto_trim(self, video_path: str, scenes: List[SceneInfo], 
                  silences: List[SilenceInfo], min_scene_duration: float = 2.0) -> Tuple[List, List[SceneInfo]]:
        """Automatically trim video keeping good scenes"""
        clip = VideoFileClip(video_path)
        
        kept_scenes = []
        for scene in scenes:
            # Skip short scenes
            if scene.duration < min_scene_duration:
                scene.kept = False
                continue
            
            # Check if scene overlaps with silence
            is_silent = False
            for silence in silences:
                if (scene.start <= silence.end and scene.end >= silence.start):
                    overlap = min(scene.end, silence.end) - max(scene.start, silence.start)
                    if overlap > scene.duration * 0.5:  # More than 50% overlap
                        is_silent = True
                        break
            
            if not is_silent:
                scene.kept = True
                kept_scenes.append(scene)
            else:
                scene.kept = False
        
        # Create subclips from kept scenes
        if not kept_scenes:
            kept_scenes = [SceneInfo(start=0, end=min(30, clip.duration), duration=min(30, clip.duration))]
        
        subclips = [clip.subclip(s.start, s.end) for s in kept_scenes]
        
        return subclips, kept_scenes
    
    def apply_transitions(self, clips: List, transition_duration: float = 0.5):
        """Add smooth transitions between clips"""
        if len(clips) <= 1:
            return clips[0] if clips else None
        
        # Apply fade out to all clips except last
        for i in range(len(clips) - 1):
            clips[i] = fadeout(clips[i], transition_duration)
        
        # Apply fade in to all clips except first
        for i in range(1, len(clips)):
            clips[i] = fadein(clips[i], transition_duration)
        
        return concatenate_videoclips(clips, method="compose")
    
    def apply_color_grading(self, clip, brightness: float = 1.2, 
                           contrast: float = 1.1, saturation: float = 1.3):
        """Apply color grading for product videos"""
        def color_adjust(get_frame, t):
            frame = get_frame(t)
            # Convert to HSV for saturation adjustment
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)
            hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness, 0, 255)
            hsv = hsv.astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Apply contrast
            rgb = np.clip((rgb - 128) * contrast + 128, 0, 255).astype(np.uint8)
            return rgb
        
        return clip.fl(color_adjust)
    
    def add_text_overlay(self, clip, text: str, position: str = 'bottom', 
                        fontsize: int = 50):
        """Add text overlay (CTA, product info)"""
        txt_clip = TextClip(text, fontsize=fontsize, color='white', 
                           font='Arial-Bold', stroke_color='black', stroke_width=2)
        
        if position == 'bottom':
            txt_clip = txt_clip.set_position(('center', 0.85), relative=True)
        elif position == 'top':
            txt_clip = txt_clip.set_position(('center', 0.1), relative=True)
        else:
            txt_clip = txt_clip.set_position('center')
        
        txt_clip = txt_clip.set_duration(clip.duration)
        
        return CompositeVideoClip([clip, txt_clip])
    
    # ========== CAPTIONS ==========
    
    def generate_captions(self, video_path: str) -> Tuple[str, List[CaptionSegment]]:
        """Generate captions using Whisper"""
        if self.whisper_model is None:
            print("Loading Whisper model (this takes a minute first time)...")
            self.whisper_model = whisper.load_model("base")
        
        result = self.whisper_model.transcribe(video_path, word_timestamps=True)
        
        # Convert to CaptionSegment objects
        segments = [
            CaptionSegment(
                start=seg['start'],
                end=seg['end'],
                text=seg['text'].strip()
            )
            for seg in result['segments']
        ]
        
        # Create SRT format
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start = self._format_timestamp(seg.start)
            end = self._format_timestamp(seg.end)
            srt_lines.append(f"{i}\n{start} --> {end}\n{seg.text}\n")
        
        return "\n".join(srt_lines), segments
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def add_captions_to_video(self, clip, segments: List[CaptionSegment], 
                             fontsize: int = 40):
        """Burn captions into video"""
        def make_caption(seg: CaptionSegment):
            txt = TextClip(seg.text, fontsize=fontsize, color='white', 
                          font='Arial-Bold', bg_color='black', 
                          method='caption', size=(clip.w * 0.8, None))
            return txt.set_position(('center', 0.75), relative=True).set_start(seg.start).set_end(seg.end)
        
        caption_clips = [make_caption(seg) for seg in segments]
        
        return CompositeVideoClip([clip] + caption_clips)
    
    # ========== MULTI-FORMAT EXPORT ==========
    
    def smart_crop(self, clip, target_aspect: Tuple[int, int]):
        """Smart crop to target aspect ratio (center crop)"""
        current_aspect = clip.w / clip.h
        target_w, target_h = target_aspect
        target_aspect_ratio = target_w / target_h
        
        if abs(current_aspect - target_aspect_ratio) < 0.01:
            return clip
        
        if current_aspect > target_aspect_ratio:
            # Video is wider, crop width
            new_width = int(clip.h * target_aspect_ratio)
            x_center = clip.w // 2
            x1 = x_center - new_width // 2
            return clip.crop(x1=x1, width=new_width)
        else:
            # Video is taller, crop height
            new_height = int(clip.w / target_aspect_ratio)
            y_center = clip.h // 2
            y1 = y_center - new_height // 2
            return clip.crop(y1=y1, height=new_height)
    
    def export_video(self, clip, output_path: str, resolution: Tuple[int, int], 
                    aspect_ratio: Tuple[int, int]) -> ExportInfo:
        """Export single video format"""
        # Crop to aspect ratio
        cropped = self.smart_crop(clip, aspect_ratio)
        
        # Resize
        resized = cropped.resize(height=resolution[1])
        
        # Export
        resized.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=30,
            preset='medium',
            threads=4,
            logger=None  # Suppress moviepy output
        )
        
        file_size = os.path.getsize(output_path)
        
        return ExportInfo(
            format_name=os.path.basename(output_path).split('_')[-1].replace('.mp4', ''),
            filepath=output_path,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            file_size=file_size
        )
    
    def export_multi_format(self, clip, output_base: str, 
                           formats: List[str]) -> List[ExportInfo]:
        """Export video in multiple formats"""
        presets = {
            'youtube': {'aspect': (16, 9), 'resolution': (1280, 720)},
            'instagram_story': {'aspect': (9, 16), 'resolution': (1080, 1920)},
            'instagram_feed': {'aspect': (1, 1), 'resolution': (1080, 1080)},
            'square': {'aspect': (1, 1), 'resolution': (1080, 1080)},
            'tiktok': {'aspect': (9, 16), 'resolution': (1080, 1920)},
            'facebook': {'aspect': (16, 9), 'resolution': (1280, 720)}
        }
        
        exports = []
        
        for fmt in formats:
            if fmt not in presets:
                continue
            
            preset = presets[fmt]
            output_path = f"{output_base}_{fmt}.mp4"
            
            print(f"  Exporting {fmt}...")
            
            export_info = self.export_video(
                clip, 
                output_path, 
                preset['resolution'], 
                preset['aspect']
            )
            exports.append(export_info)
        
        return exports

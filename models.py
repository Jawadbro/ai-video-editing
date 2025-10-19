from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
import json

class ProcessingStage(Enum):
    QUEUED = "queued"
    UPLOADED = "uploaded"
    ANALYZING = "analyzing"
    DETECTING_SCENES = "detecting_scenes"
    DETECTING_SILENCES = "detecting_silences"
    EDITING = "editing"
    APPLYING_TRANSITIONS = "applying_transitions"
    COLOR_GRADING = "color_grading"
    ADDING_OVERLAYS = "adding_overlays"
    CAPTIONING = "captioning"
    EXPORTING = "exporting"
    COMPLETE = "complete"
    FAILED = "failed"

class ErrorSeverity(Enum):
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ProcessingError:
    stage: str
    message: str
    severity: ErrorSeverity
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class VideoMetadata:
    duration: float
    fps: float
    width: int
    height: int
    resolution: str
    has_audio: bool
    file_size: int
    codec: str = "unknown"

@dataclass
class SceneInfo:
    start: float
    end: float
    duration: float
    kept: bool = True

@dataclass
class SilenceInfo:
    start: float
    end: float
    duration: float

@dataclass
class CaptionSegment:
    start: float
    end: float
    text: str

@dataclass
class ExportInfo:
    format_name: str
    filepath: str
    resolution: tuple
    aspect_ratio: tuple
    file_size: int = 0

@dataclass
class ProcessingOptions(dict):
    """
    Flexible options container for agent output.
    Accepts any keys, provides defaults for known ones.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set defaults for known keys if not present
        self.setdefault('color_grading', True)
        self.setdefault('brightness', 1.2)
        self.setdefault('contrast', 1.1)
        self.setdefault('saturation', 1.3)
        self.setdefault('captions', True)
        self.setdefault('burn_captions', False)
        self.setdefault('caption_fontsize', 40)
        self.setdefault('text_overlay', None)
        self.setdefault('text_position', 'bottom')
        self.setdefault('text_fontsize', 50)
        self.setdefault('formats', ['youtube', 'instagram_story', 'square'])
        self.setdefault('min_scene_duration', 2.0)
        self.setdefault('transition_duration', 0.5)
        self.setdefault('detect_black_frames', True)
        self.setdefault('detect_audio_clipping', True)
        # Accepts any extra keys (e.g., color_tone, special_effects, etc.)

    def get_known(self, key):
        # Helper to get known keys with default
        return self.get(key, None)


@dataclass
class VideoState:
    video_id: str
    filename: str
    input_path: str
    stage: ProcessingStage
    progress: int = 0
    metadata: Optional[VideoMetadata] = None
    thumbnail_path: Optional[str] = None
    scenes: List[SceneInfo] = field(default_factory=list)
    silences: List[SilenceInfo] = field(default_factory=list)
    kept_scenes: List[SceneInfo] = field(default_factory=list)
    captions: List[CaptionSegment] = field(default_factory=list)
    srt_path: Optional[str] = None
    exports: List[ExportInfo] = field(default_factory=list)
    has_black_frames: bool = False
    has_audio_clipping: bool = False
    quality_warnings: List[str] = field(default_factory=list)
    errors: List[ProcessingError] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['stage'] = self.stage.value
        return data
    
    def update_progress(self, stage: ProcessingStage, progress: int):
        self.stage = stage
        self.progress = progress
        self.updated_at = datetime.now().isoformat()
    
    def add_error(self, stage: str, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR):
        error = ProcessingError(stage=stage, message=message, severity=severity)
        self.errors.append(error)
    
    def mark_complete(self):
        self.stage = ProcessingStage.COMPLETE
        self.progress = 100
        self.completed_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def mark_failed(self, error_message: str):
        self.stage = ProcessingStage.FAILED
        self.add_error("pipeline", error_message, ErrorSeverity.CRITICAL)
        self.updated_at = datetime.now().isoformat()

class StateManager:
    def __init__(self, state_dir: str = "states"):
        self.state_dir = state_dir
        self.states: Dict[str, VideoState] = {}
        import os
        os.makedirs(state_dir, exist_ok=True)
    
    def create_state(self, video_id: str, filename: str, input_path: str) -> VideoState:
        state = VideoState(
            video_id=video_id,
            filename=filename,
            input_path=input_path,
            stage=ProcessingStage.QUEUED
        )
        self.states[video_id] = state
        self.save_state(video_id)
        return state
    
    def get_state(self, video_id: str) -> Optional[VideoState]:
        if video_id in self.states:
            return self.states[video_id]
        return self.load_state(video_id)
    
    def save_state(self, video_id: str):
        if video_id not in self.states:
            return
        state = self.states[video_id]
        filepath = f"{self.state_dir}/{video_id}.json"
        with open(filepath, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
    
    def load_state(self, video_id: str) -> Optional[VideoState]:
        filepath = f"{self.state_dir}/{video_id}.json"
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                state = VideoState(
                    video_id=data['video_id'],
                    filename=data['filename'],
                    input_path=data['input_path'],
                    stage=ProcessingStage(data['stage']),
                    progress=data['progress']
                )
                self.states[video_id] = state
                return state
        except FileNotFoundError:
            return None

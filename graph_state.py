"""
LangGraph State Schema - Production Ready
All fields are serializable for checkpointing
"""

from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages


class VideoEditingState(TypedDict):
    """
    Complete state for video editing workflow
    All fields are serializable - no MoviePy objects stored
    """
    
    # ========== INPUT ==========
    video_id: str
    filename: str
    input_path: str
    user_instruction: str
    
    # ========== AGENT PLANNING ==========
    agent_plan: Optional[dict]
    editing_options: Optional[dict]
    
    # ========== VIDEO ANALYSIS ==========
    metadata: Optional[dict]
    thumbnail_path: Optional[str]
    scenes: List[dict]
    silences: List[dict]
    kept_scenes: List[dict]
    
    # ========== QUALITY ASSURANCE ==========
    quality_score: float  # 0-100
    quality_issues: List[str]
    needs_enhancement: bool
    
    # ========== PROCESSING RESULTS ==========
    # Store paths instead of objects (serializable)
    temp_edited_path: Optional[str]  # Path to temporary edited video
    captions: List[dict]
    srt_path: Optional[str]
    exports: List[dict]
    
    # ========== WORKFLOW CONTROL ==========
    current_stage: str
    progress: int  # 0-100
    retry_count: int
    max_retries: int
    errors: Annotated[list, add_messages]  # LangGraph message tracking
    
    # ========== HUMAN-IN-THE-LOOP ==========
    requires_approval: bool
    approved: bool

"""
LangGraph Orchestrator - The Agentic Workflow Engine
Coordinates all nodes with conditional logic and state management
Production ready with proper serialization
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from graph_state import VideoEditingState
from graph_nodes import (
    agent_planning_node,
    validate_input_node,
    analyze_video_node,
    quality_check_node,
    enhance_video_node,
    edit_video_node,
    generate_captions_node,
    export_node,
    error_handler_node
)


class LangGraphOrchestrator:
    """
    Truly agentic orchestrator using LangGraph
    
    Features:
    - Conditional branching based on quality checks
    - Retry logic for failed operations
    - State persistence with checkpointing
    - Graph-based workflow (not linear)
    - Production-ready serialization
    """
    
    def __init__(self):
        # Create the workflow graph
        self.workflow = StateGraph(VideoEditingState)
        
        # ========== ADD NODES ==========
        self.workflow.add_node("agent_planning", agent_planning_node)
        self.workflow.add_node("validate", validate_input_node)
        self.workflow.add_node("analyze", analyze_video_node)
        self.workflow.add_node("quality_check", quality_check_node)
        self.workflow.add_node("enhance", enhance_video_node)
        self.workflow.add_node("edit", edit_video_node)
        self.workflow.add_node("captions", generate_captions_node)
        self.workflow.add_node("export", export_node)
        self.workflow.add_node("error_handler", error_handler_node)
        
        # ========== DEFINE WORKFLOW EDGES ==========
        
        # Entry point
        self.workflow.set_entry_point("agent_planning")
        
        # Agent planning → Validation
        self.workflow.add_edge("agent_planning", "validate")
        
        # Conditional: Validation success or failure
        self.workflow.add_conditional_edges(
            "validate",
            self._should_continue_after_validation,
            {
                "continue": "analyze",
                "error": "error_handler"
            }
        )
        
        # Analysis → Quality Check
        self.workflow.add_edge("analyze", "quality_check")
        
        # Conditional: Good quality or needs enhancement
        self.workflow.add_conditional_edges(
            "quality_check",
            self._should_enhance,
            {
                "enhance": "enhance",
                "skip": "edit"
            }
        )
        
        # Enhancement → Edit
        self.workflow.add_edge("enhance", "edit")
        
        # Edit → Captions
        self.workflow.add_edge("edit", "captions")
        
        # Captions → Export
        self.workflow.add_edge("captions", "export")
        
        # Export → END
        self.workflow.add_edge("export", END)
        
        # Error handling: Retry or fail
        self.workflow.add_conditional_edges(
            "error_handler",
            self._should_retry,
            {
                "retry": "validate",
                "fail": END
            }
        )
        
        # ========== COMPILE WITH CHECKPOINTING ==========
        # Now works because we store paths, not objects
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    # ========== CONDITIONAL LOGIC FUNCTIONS ==========
    
    def _should_continue_after_validation(self, state: VideoEditingState) -> str:
        """
        Decide if validation passed
        Returns: "continue" or "error"
        """
        if state.get('current_stage') == 'failed':
            return "error"
        return "continue"
    
    def _should_enhance(self, state: VideoEditingState) -> str:
        """
        Decide if video needs quality enhancement
        Returns: "enhance" or "skip"
        """
        if state.get('needs_enhancement', False):
            return "enhance"
        return "skip"
    
    def _should_retry(self, state: VideoEditingState) -> str:
        """
        Decide if should retry after error
        Returns: "retry" or "fail"
        """
        if state.get('current_stage') == 'retrying':
            return "retry"
        return "fail"
    
    # ========== MAIN EXECUTION METHOD ==========
    
    def process_video(self, video_id: str, filename: str, input_path: str, 
                     user_instruction: str) -> VideoEditingState:
        """
        Execute the agentic workflow with LangGraph
        
        Args:
            video_id: Unique identifier for this job
            filename: Original filename
            input_path: Path to input video
            user_instruction: Natural language instruction from user
        
        Returns:
            Final VideoEditingState with all results
        """
        
        # ========== INITIALIZE STATE (UPDATED) ==========
        initial_state = VideoEditingState(
            # Input
            video_id=video_id,
            filename=filename,
            input_path=input_path,
            user_instruction=user_instruction,
            
            # Agent planning
            agent_plan=None,
            editing_options=None,
            
            # Video analysis
            metadata=None,
            thumbnail_path=None,
            scenes=[],
            silences=[],
            kept_scenes=[],
            
            # Quality
            quality_score=0.0,
            quality_issues=[],
            needs_enhancement=False,
            
            # Results (UPDATED - no clip object, just path)
            temp_edited_path=None,  # Changed from edited_clip
            captions=[],
            srt_path=None,
            exports=[],
            
            # Workflow control
            current_stage='initialized',
            progress=0,
            retry_count=0,
            max_retries=2,
            errors=[],
            
            # Human-in-loop
            requires_approval=False,
            approved=True
        )
        
        # ========== RUN THE GRAPH ==========
        config = {"configurable": {"thread_id": video_id}}
        
        print("\n" + "="*70)
        print("🚀 LANGGRAPH AGENTIC WORKFLOW STARTING")
        print("="*70)
        print(f"Video ID: {video_id}")
        print(f"Instruction: \"{user_instruction}\"")
        print("="*70)
        
        final_state = None
        
        # Stream through the graph execution
        try:
            for state_update in self.app.stream(initial_state, config):
                # Get the node that just executed
                node_name = list(state_update.keys())[0]
                node_state = state_update[node_name]
                
                # Display progress
                progress = node_state.get('progress', 0)
                stage = node_state.get('current_stage', 'unknown')
                
                print(f"Progress: {progress}% | Stage: {stage}")
                
                # Store final state
                final_state = node_state
                
                # Check for errors
                if stage == 'failed':
                    print("⚠️ Workflow encountered an error")
                    break
        
        except Exception as e:
            print(f"\n⚠️ Workflow error: {str(e)}")
            if final_state:
                final_state['current_stage'] = 'failed'
                final_state['errors'] = final_state.get('errors', []) + [str(e)]
            else:
                final_state = initial_state
                final_state['current_stage'] = 'failed'
                final_state['errors'] = [str(e)]
        
        print("\n" + "="*70)
        if final_state and final_state.get('current_stage') == 'completed':
            print("✅ WORKFLOW COMPLETED SUCCESSFULLY")
        else:
            print("❌ WORKFLOW FAILED")
            errors = final_state.get('errors', []) if final_state else []
            if errors:
                print("\nErrors:")
                for err in errors:
                    print(f"  - {err}")
        print("="*70 + "\n")
        
        return final_state
    
    # ========== UTILITY METHODS ==========
    
    def get_graph_visualization(self):
        """
        Get a visual representation of the workflow graph
        (Optional - requires IPython)
        """
        try:
            from IPython.display import Image, display
            display(Image(self.app.get_graph().draw_mermaid_png()))
        except Exception as e:
            print("Graph visualization not available")
            print("(Requires IPython - install with: pip install ipython)")
    
    def list_checkpoints(self, video_id: str):
        """
        List all saved checkpoints for a video processing job
        Useful for debugging and recovery
        """
        config = {"configurable": {"thread_id": video_id}}
        try:
            checkpoints = list(self.app.get_state_history(config))
            return checkpoints
        except Exception as e:
            print(f"Could not retrieve checkpoints: {e}")
            return []
    
    def resume_from_checkpoint(self, video_id: str, checkpoint_id: str):
        """
        Resume processing from a specific checkpoint
        Useful for recovering from failures
        """
        config = {
            "configurable": {
                "thread_id": video_id,
                "checkpoint_id": checkpoint_id
            }
        }
        
        print(f"\n🔄 Resuming from checkpoint: {checkpoint_id}")
        
        # Resume execution
        final_state = None
        for state in self.app.stream(None, config):
            node_name = list(state.keys())[0]
            node_state = state[node_name]
            progress = node_state.get('progress', 0)
            print(f"Progress: {progress}% | Stage: {node_state.get('current_stage')}")
            final_state = node_state
        
        return final_state
    
    def get_workflow_status(self, video_id: str):
        """
        Get current status of a workflow
        """
        config = {"configurable": {"thread_id": video_id}}
        try:
            state = self.app.get_state(config)
            return {
                'stage': state.values.get('current_stage', 'unknown'),
                'progress': state.values.get('progress', 0),
                'errors': state.values.get('errors', [])
            }
        except Exception as e:
            return {
                'stage': 'not_found',
                'progress': 0,
                'errors': [str(e)]
            }

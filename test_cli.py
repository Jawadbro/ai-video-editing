#!/usr/bin/env python3
"""
Agentic Video Editing CLI - Powered by Google Gemini + LangGraph
"""

import sys
import argparse
from langgraph_orchestrator import LangGraphOrchestrator
import uuid
import os

def main():
    parser = argparse.ArgumentParser(
        description='🤖 Agentic Video Editing - Gemini + LangGraph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Natural Language Examples:

  python test_cli.py uploads/test_video.mp4 "Make this vibrant for Instagram Stories"
  
  python test_cli.py uploads/test_video.mp4 "Professional YouTube video with captions"
  
  python test_cli.py uploads/test_video.mp4 "Quick TikTok with Shop Now text"
  
  python test_cli.py uploads/test_video.mp4 "Cinematic product showcase"

Powered by: Google Gemini (FREE) + LangGraph
        """
    )
    
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('instruction', help='Natural language instruction')
    parser.add_argument('--api-key', help='Google API key (or set in .env)')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_video):
        print(f"❌ Error: File not found: {args.input_video}")
        sys.exit(1)
    
    video_id = str(uuid.uuid4())
    
    print("\n" + "="*70)
    print("🎬 AGENTIC VIDEO EDITING")
    print("="*70)
    print(f"📹 Input: {args.input_video}")
    print(f"💬 Instruction: \"{args.instruction}\"")
    print(f"🆔 Video ID: {video_id}")
    print("="*70 + "\n")
    
    try:
        # Initialize LangGraph orchestrator
        orchestrator = LangGraphOrchestrator()
        
        # Execute workflow
        final_state = orchestrator.process_video(
            video_id=video_id,
            filename=os.path.basename(args.input_video),
            input_path=args.input_video,
            user_instruction=args.instruction
        )
        
        # Display results
        if final_state and final_state.get('current_stage') == 'completed':
            print("\n📊 Results:")
            
            # Metadata
            metadata = final_state.get('metadata', {})
            if metadata:
                print(f"\nVideo Info:")
                print(f"  Duration: {metadata.get('duration', 0):.2f}s")
                print(f"  Resolution: {metadata.get('resolution', 'N/A')}")
            
            # Scenes
            kept_scenes = final_state.get('kept_scenes', [])
            if kept_scenes:
                print(f"\nEditing:")
                print(f"  Kept {len(kept_scenes)} scenes")
            
            # Captions
            captions = final_state.get('captions', [])
            if captions:
                print(f"\nCaptions:")
                print(f"  {len(captions)} segments")
                print(f"  SRT: {final_state.get('srt_path', 'N/A')}")
            
            # Exports
            exports = final_state.get('exports', [])
            if exports:
                print(f"\n📦 Exported Videos:")
                total_size = 0
                for exp in exports:
                    size_mb = exp['file_size'] / (1024*1024)
                    total_size += size_mb
                    print(f"  ✓ {exp['format_name'].upper()}: {exp['filepath']}")
                    print(f"    {exp['resolution'][0]}x{exp['resolution'][1]} ({size_mb:.2f} MB)")
                print(f"\n💾 Total: {total_size:.2f} MB")
            
            print("\n" + "="*70)
            print("🎉 Check the 'exports' folder!")
            print("="*70 + "\n")
            
            return 0
        else:
            print("\n❌ Processing failed")
            errors = final_state.get('errors', []) if final_state else []
            if errors:
                print("Errors:")
                for err in errors:
                    print(f"  - {err}")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Cancelled by user")
        return 1
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

"""
Flask API for Agentic Video Editing
Powered by LangGraph + Gemini
"""

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import uuid
from langgraph_orchestrator import LangGraphOrchestrator
import threading
from datetime import datetime

# Tell Flask where your static files are (default is 'static')
app = Flask(__name__, static_folder='static')

# Configuration
UPLOAD_FOLDER = 'uploads'
EXPORT_FOLDER = 'exports'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# Initialize orchestrator
orchestrator = LangGraphOrchestrator()

# Job tracking dictionary
jobs = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Serve your web UI index.html at root URL
@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/v1/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'agentic-video-editor',
        'version': '2.0',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/v1/upload', methods=['POST'])
def upload_video():
    """
    Upload video and start processing
    
    Body:
    - video: Video file
    - instruction: Natural language instruction
    
    Returns:
    - job_id: Unique job identifier
    """
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    if 'instruction' not in request.form:
        return jsonify({'error': 'No instruction provided'}), 400
    
    file = request.files['video']
    instruction = request.form['instruction']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, mov, avi, mkv'}), 400
    
    video_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"{video_id}_{filename}")
    file.save(filepath)
    
    jobs[video_id] = {
        'status': 'processing',
        'instruction': instruction,
        'filename': filename,
        'filepath': filepath,
        'progress': 0,
        'created_at': datetime.utcnow().isoformat(),
        'result': None,
        'error': None
    }
    
    thread = threading.Thread(
        target=process_video_async,
        args=(video_id, filename, filepath, instruction)
    )
    thread.start()
    
    return jsonify({
        'job_id': video_id,
        'status': 'processing',
        'message': 'Video uploaded successfully. Processing started.'
    }), 202


def process_video_async(video_id, filename, filepath, instruction):
    """Process video in background thread"""
    try:
        final_state = orchestrator.process_video(
            video_id=video_id,
            filename=filename,
            input_path=filepath,
            user_instruction=instruction
        )
        
        if final_state.get('current_stage') == 'completed':
            jobs[video_id]['status'] = 'completed'
            jobs[video_id]['progress'] = 100
            jobs[video_id]['result'] = {
                'exports': final_state.get('exports', []),
                'metadata': final_state.get('metadata', {}),
                'kept_scenes': final_state.get('kept_scenes', []),
                'captions': final_state.get('captions', []),
                'srt_path': final_state.get('srt_path')
            }
        else:
            jobs[video_id]['status'] = 'failed'
            jobs[video_id]['error'] = final_state.get('errors', ['Unknown error'])
    
    except Exception as e:
        jobs[video_id]['status'] = 'failed'
        jobs[video_id]['error'] = str(e)


@app.route('/api/v1/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])


@app.route('/api/v1/download/<job_id>/<format_name>', methods=['GET'])
def download_video(job_id, format_name):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400
    
    exports = job['result']['exports']
    export = next((e for e in exports if e['format_name'] == format_name), None)
    
    if not export:
        return jsonify({'error': f'Format {format_name} not found'}), 404
    
    filepath = export['filepath']
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(filepath, as_attachment=True)


@app.route('/api/v1/jobs', methods=['GET'])
def list_jobs():
    return jsonify({
        'jobs': [
            {
                'job_id': job_id,
                'status': job['status'],
                'instruction': job['instruction'],
                'created_at': job['created_at']
            }
            for job_id, job in jobs.items()
        ]
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 AGENTIC VIDEO EDITING API")
    print("="*70)
    print("Powered by: LangGraph + Gemini")
    print("Endpoints:")
    print("  POST   /api/v1/upload          - Upload and process video")
    print("  GET    /api/v1/status/<job_id> - Check job status")
    print("  GET    /api/v1/download/<job_id>/<format> - Download result")
    print("  GET    /api/v1/jobs            - List all jobs")
    print("  GET    /api/v1/health          - Health check")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

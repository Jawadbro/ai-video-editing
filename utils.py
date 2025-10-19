import cv2
import numpy as np
from typing import List, Tuple
import os

class QualityAssurance:
    @staticmethod
    def detect_black_frames(video_path: str, threshold: int = 10) -> List[Tuple[float, float]]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        black_sequences = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < threshold:
                black_sequences.append((frame_count / fps, (frame_count + 1) / fps))
            frame_count += 1
        
        cap.release()
        return black_sequences
    
    @staticmethod
    def validate_video_file(filepath: str) -> Tuple[bool, str]:
        if not os.path.exists(filepath):
            return False, "File does not exist"
        if os.path.getsize(filepath) == 0:
            return False, "File is empty"
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return False, "Cannot open video file"
        ret, _ = cap.read()
        cap.release()
        if not ret:
            return False, "Cannot read video frames"
        return True, "Valid"

class FileManager:
    @staticmethod
    def ensure_directories():
        dirs = ['uploads', 'exports', 'temp', 'states']
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    @staticmethod
    def get_file_size(filepath: str) -> int:
        return os.path.getsize(filepath) if os.path.exists(filepath) else 0

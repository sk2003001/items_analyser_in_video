# app.py - Main Flask Application
from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from collections import defaultdict, Counter
import tempfile
import uuid
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  
model = YOLO('yolov8n.pt')  

ENVIRONMENT_CLASSIFIERS = {
    'home': {
        'keywords': ['sofa', 'couch', 'bed', 'chair', 'dining table', 'tv', 'refrigerator', 
                    'microwave', 'oven', 'sink', 'toilet', 'bathtub', 'potted plant'],
        'threshold': 3
    },
    'shop': {
        'keywords': ['bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 'broccoli', 
                    'carrot', 'pizza', 'donut', 'cake', 'laptop', 'mouse', 'keyboard',
                    'cell phone', 'book', 'clock', 'scissors', 'teddy bear', 'hair drier',
                    'toothbrush'],
        'threshold': 5
    },
    'office': {
        'keywords': ['laptop', 'mouse', 'keyboard', 'cell phone', 'book', 'chair',
                    'dining table', 'clock', 'tv'],
        'threshold': 4
    }
}


ITEM_CATEGORIES = {
    'furniture': ['sofa', 'couch', 'chair', 'dining table', 'bed'],
    'appliances': ['tv', 'laptop', 'refrigerator', 'microwave', 'oven', 'hair drier'],
    'electronics': ['tv', 'laptop', 'mouse', 'keyboard', 'cell phone', 'remote'],
    'kitchenware': ['bottle', 'cup', 'bowl', 'fork', 'knife', 'spoon'],
    'food_items': ['banana', 'apple', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'cake'],
    'decor': ['potted plant', 'vase', 'clock', 'teddy bear'],
    'shop_infrastructure': ['refrigerator', 'dining table'], 
    'inventory': ['bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 'book']
}

class VideoAnalyzer:
    def __init__(self, model):
        self.model = model
        
    def extract_frames(self, video_path, max_frames=30):
        """Extract frames from video for analysis"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        
        if frame_count <= max_frames:
            interval = 1
        else:
            interval = frame_count // max_frames
            
        frame_idx = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % interval == 0:
                frames.append(frame)
            frame_idx += 1
            
        cap.release()
        return frames
    
    def analyze_frame(self, frame):
        """Analyze a single frame and return detections"""
        results = self.model(frame, conf=0.3)  
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': box.xyxy[0].tolist()
                    })
        
        return detections
    
    def analyze_video(self, video_path):
        """Analyze entire video and return comprehensive results"""
        frames = self.extract_frames(video_path)
        all_detections = []
        frame_results = []
        
        for i, frame in enumerate(frames):
            detections = self.analyze_frame(frame)
            all_detections.extend(detections)
            frame_results.append({
                'frame_index': i,
                'detections': detections
            })
        
        
        item_counts = Counter([det['class'] for det in all_detections])
        
       
        item_confidences = defaultdict(list)
        for det in all_detections:
            item_confidences[det['class']].append(det['confidence'])
        
        avg_confidences = {
            item: sum(confidences) / len(confidences)
            for item, confidences in item_confidences.items()
        }
        
       
        environment = self.classify_environment(item_counts)
        
       
        categorized_items = self.categorize_items(item_counts, environment)
        
        return {
            'environment': environment,
            'total_frames_analyzed': len(frames),
            'item_counts': dict(item_counts),
            'item_confidences': avg_confidences,
            'categorized_items': categorized_items,
            'frame_results': frame_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def classify_environment(self, item_counts):
        """Classify environment based on detected items"""
        scores = {}
        
        for env_type, config in ENVIRONMENT_CLASSIFIERS.items():
            score = 0
            matched_items = []
            
            for keyword in config['keywords']:
                if keyword in item_counts:
                    score += item_counts[keyword]
                    matched_items.append(keyword)
            
            scores[env_type] = {
                'score': score,
                'matched_items': matched_items,
                'threshold_met': len(matched_items) >= config['threshold']
            }
        
        
        best_env = 'unknown'
        best_score = 0
        
        for env_type, data in scores.items():
            if data['threshold_met'] and data['score'] > best_score:
                best_env = env_type
                best_score = data['score']
        
        return {
            'type': best_env,
            'confidence_scores': scores
        }
    
    def categorize_items(self, item_counts, environment):
        """Categorize detected items based on environment"""
        categorized = defaultdict(dict)
        
        for item, count in item_counts.items():
            
            item_category = 'other'
            for category, items in ITEM_CATEGORIES.items():
                if item in items:
                    item_category = category
                    break
            
            categorized[item_category][item] = count
        
        return dict(categorized)

analyzer = VideoAnalyzer(model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    temp_dir = tempfile.gettempdir()
    video_id = str(uuid.uuid4())
    video_path = os.path.join(temp_dir, f"{video_id}_{video_file.filename}")
    video_file.save(video_path)
    
    try:
        results = analyzer.analyze_video(video_path)
        
        
        results_path = os.path.join(temp_dir, f"results_{video_id}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
       
        os.remove(video_path)
        
        return jsonify({
            'success': True,
            'results': results,
            'download_id': video_id
        })
        
    except Exception as e:
        
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({'error': str(e)}), 500

@app.route('/download/<download_id>')
def download_results(download_id):
    """Download analysis results as JSON file"""
    temp_dir = tempfile.gettempdir()
    results_path = os.path.join(temp_dir, f"results_{download_id}.json")
    
    if not os.path.exists(results_path):
        return jsonify({'error': 'Results not found'}), 404
    
    return send_file(results_path, 
                    as_attachment=True, 
                    download_name=f"video_analysis_{download_id}.json",
                    mimetype='application/json')

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
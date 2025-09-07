
import sys
import os
import tempfile
import numpy as np
import cv2
from datetime import datetime

def test_imports():
    """Test if all required libraries can be imported"""
    try:
        import flask
        print(f"✓ Flask {flask.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"✗ YOLO import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True

def test_yolo_model():
    """Test YOLO model loading and basic inference"""
    try:
        from ultralytics import YOLO
        print("Loading YOLO model...")
        model = YOLO('yolov8n.pt')
        print("✓ YOLO model loaded successfully")
        
        
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
      
        results = model(test_image, verbose=False)
        print(f"✓ YOLO inference successful, detected {len(results[0].boxes) if results[0].boxes is not None else 0} objects")
        
        return model
    except Exception as e:
        print(f"✗ YOLO model test failed: {e}")
        return None

def create_test_video():
    """Create a simple test video for testing purposes"""
    try:
        
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, "test_video.mp4")
        
     
        width, height = 640, 480
        fps = 10
        duration = 3  
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
       
        for frame_num in range(fps * duration):
            
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            
            cv2.rectangle(frame, (50, 50), (200, 150), (0, 255, 0), -1)  
            cv2.rectangle(frame, (300, 200), (500, 350), (255, 0, 0), -1)  
            cv2.rectangle(frame, (100, 300), (250, 400), (0, 0, 255), -1)  
            
           
            cv2.putText(frame, f'Frame {frame_num}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✓ Test video created: {video_path}")
        return video_path
        
    except Exception as e:
        print(f"✗ Test video creation failed: {e}")
        return None

def test_video_analysis(model, video_path):
    """Test video analysis functionality"""
    try:
       
        sys.path.append('.')
        from app import VideoAnalyzer
        
        analyzer = VideoAnalyzer(model)
        
       
        frames = analyzer.extract_frames(video_path, max_frames=10)
        print(f"✓ Extracted {len(frames)} frames from test video")
        

        if frames:
            detections = analyzer.analyze_frame(frames[0])
            print(f"✓ Frame analysis completed, {len(detections)} detections")
        
      
        results = analyzer.analyze_video(video_path)
        print("✓ Full video analysis completed")
        print(f"  - Environment: {results['environment']['type']}")
        print(f"  - Items detected: {len(results['item_counts'])}")
        print(f"  - Frames analyzed: {results['total_frames_analyzed']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Video analysis test failed: {e}")
        return False

def test_flask_app():
    """Test Flask app initialization"""
    try:
        sys.path.append('.')
        from app import app
        
        with app.test_client() as client:
            
            response = client.get('/health')
            if response.status_code == 200:
                print("✓ Flask app health check passed")
                return True
            else:
                print(f"✗ Flask app health check failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"✗ Flask app test failed: {e}")
        return False

def cleanup(video_path):
    """Clean up test files"""
    try:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            print("✓ Test files cleaned up")
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Video Analysis App - Setup Validation Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    
    print("1. Testing imports...")
    if not test_imports():
        print("✗ Import test failed. Please install required packages.")
        return
    print()
    
  
    print("2. Testing YOLO model...")
    model = test_yolo_model()
    if model is None:
        print("✗ YOLO model test failed.")
        return
    print()
    
    
    print("3. Creating test video...")
    video_path = create_test_video()
    if video_path is None:
        print("✗ Test video creation failed.")
        return
    print()
    
   
    print("4. Testing video analysis...")
    if not test_video_analysis(model, video_path):
        cleanup(video_path)
        return
    print()
    

    print("5. Testing Flask app...")
    if not test_flask_app():
        cleanup(video_path)
        return
    print()
    
   
    cleanup(video_path)
    
    print("=" * 60)
    print("🎉 ALL TESTS PASSED! Your setup is ready.")
    print("You can now run 'python app.py' to start the application.")
    print("=" * 60)

if __name__ == "__main__":
    main()
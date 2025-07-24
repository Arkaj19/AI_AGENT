
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import numpy as np
import json
from collections import defaultdict, deque
from typing import Dict
import io
import os
import warnings
import gc
warnings.filterwarnings("ignore", category=FutureWarning)
app = FastAPI(title="Crowd Monitoring API")
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize YOLO model with better error handling
model = None
try:
    from ultralytics import YOLO
    import torch
    
    # Mock the function for compatibility
    if not hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals = lambda x: None
    
    model = YOLO("yolov8n.pt")
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print(f"❌ YOLO failed to load: {e}")
    print("Will use fallback detection method")
    model = None
# Add the fallback detection function here
def detect_persons_fallback(frame):
    """Lightweight fallback person detection using OpenCV"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        edges = cv2.Canny(blurred, 30, 80)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        person_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 800 < area < 8000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                if 1.2 < aspect_ratio < 4.0:
                    person_count += 1
        
        return min(person_count, 30)
    except Exception as e:
        print(f"Fallback detection error: {e}")
        return 5  # Default fallback
# Rest of your code continues...
feeds_data = {}
# analytics_data = defaultdict(lambda: deque(maxlen=100))
analytics_data = defaultdict(lambda: deque(maxlen=50))
alert_threshold = 20
congestion_threshold = 15
# CCTV Feed Configuration
CCTV_FEEDS = {
    "feed_1": {
        "name": "Main Entrance",
        "url": "https://res.cloudinary.com/do5orgcfk/video/upload/v1753219938/main_gate_r2mv6f.mp4",
        "location": {"lat": 28.6139, "lng": 77.2090},
        "area": "entrance",
        "max_capacity": 50
    },
    "feed_2": {
        "name": "Mall Stage",
        "url": "https://res.cloudinary.com/do5orgcfk/video/upload/v1753219924/mall_stair_x2j0y4.mp4",
        "location": {"lat": 28.6140, "lng": 77.2091},
        "area": "stage",
        "max_capacity": 100
    },
    "feed_3": {
        "name": "Red Street Road",
        "url": "https://res.cloudinary.com/do5orgcfk/video/upload/v1753219928/red_street_movzzb.mp4",
        "location": {"lat": 28.6141, "lng": 77.2092},
        "area": "food_court",
        "max_capacity": 30
    },
    "feed_4": {
        "name": "Exit Gate",
        "url": "https://res.cloudinary.com/do5orgcfk/video/upload/v1753219934/show_road_eqck6y.mp4",
        "location": {"lat": 28.6142, "lng": 77.2093},
        "area": "exit_a",
        "max_capacity": 25
    },
    "feed_5": {
        "name": "Subway",
        "url": "https://res.cloudinary.com/do5orgcfk/video/upload/v1753219926/subway_egn5rh.mp4",
        "location": {"lat": 28.6143, "lng": 77.2094},
        "area": "exit_b",
        "max_capacity": 25
    }
}
########### New Code ##############
# Function to validate video files exist
def validate_video_files():
    """Check if all video files exist and are accessible"""
    missing_files = []
    valid_files = []
    
    for feed_id, config in CCTV_FEEDS.items():
        video_path = config['url']
        
        if os.path.exists(video_path):
            # Test if OpenCV can open the file
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                # Get video properties for verification
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                valid_files.append({
                    "feed_id": feed_id,
                    "path": video_path,
                    "frame_count": frame_count,
                    "fps": fps,
                    "resolution": f"{width}x{height}"
                })
                cap.release()
            else:
                missing_files.append(f"{feed_id}: Cannot open {video_path}")
        else:
            missing_files.append(f"{feed_id}: File not found {video_path}")
    
    return valid_files, missing_files
# Add this to your FastAPI app startup
def check_video_setup():
    """Check video setup on startup"""
    print("Validating video files...")
    valid_files, missing_files = validate_video_files()
    
    if missing_files:
        print("❌ Missing or invalid video files:")
        for missing in missing_files:
            print(f"   {missing}")
    
    if valid_files:
        print("✅ Valid video files found:")
        for valid in valid_files:
            print(f"   {valid['feed_id']}: {valid['resolution']} @ {valid['fps']:.1f}fps ({valid['frame_count']} frames)")
    
    return len(valid_files), len(missing_files)
########### End of New Code ##############
# class FeedProcessor:
#     def __init__(self, feed_id, feed_config):  # FIXED: Double underscores
#         self.feed_id = feed_id
#         self.config = feed_config
#         self.cap = None
#         self.running = False
#         self.current_count = 0
#         self.last_logged = time.time()
#         self.log_interval = 5  # seconds
#         self.retry_count = 0
#         self.max_retries = 3


#     def _process_feed(self):
#         try:
#             self.cap = cv2.VideoCapture(self.config['url'])
            
#             # Set properties for better streaming
#             self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
#             if not self.cap.isOpened():
#                 print(f"Failed to open video stream for {self.feed_id}")
#                 return
#             while self.running:
#                 ret, frame = self.cap.read()
                
#                 if not ret:
#                     # Video ended or connection issue - restart the video
#                     print(f"Video ended or failed to read from {self.feed_id}, restarting...")
                    
#                     # Release and reinitialize
#                     if self.cap:
#                         self.cap.release()
                    
#                     time.sleep(1)  # Brief pause before restart
#                     self.cap = cv2.VideoCapture(self.config['url'])
                    
#                     if not self.cap.isOpened():
#                         self.retry_count += 1
#                         if self.retry_count >= self.max_retries:
#                             print(f"Max retries reached for {self.feed_id}")
#                             break
#                         time.sleep(5)  # Wait longer before retry
#                         continue
#                     else:
#                         self.retry_count = 0  # Reset on success
#                         continue
#                 # Reset retry count on successful read
#                 self.retry_count = 0
#                 # Detect persons
#                 try:
#                     results = model(frame, classes=[0], verbose=False)
#                     detections = results[0].boxes
#                     person_count = len(detections) if detections is not None else 0
#                 except Exception as e:
#                     print(f"YOLO detection error for {self.feed_id}: {e}")
#                     person_count = 0
#                 # Update current count
#                 self.current_count = person_count
#                 # Log data periodically
#                 if time.time() - self.last_logged >= self.log_interval:
#                     self._log_data(person_count, frame)
#                     self.last_logged = time.time()
#                 time.sleep(0.2)  # Slightly longer delay for network streams
#         except Exception as e:
#             print(f"Error processing {self.feed_id}: {e}")
#         finally:
#             if self.cap:
#                 self.cap.release()

class FeedProcessor:
    def __init__(self, feed_id, feed_config):  # Fixed the method name
        self.feed_id = feed_id
        self.config = feed_config
        self.cap = None
        self.running = False
        self.current_count = 0
        self.last_logged = time.time()
        self.log_interval = 10  # Increased to reduce processing frequency
        self.retry_count = 0
        self.max_retries = 3
        self.frame_skip = 3  # Process every 3rd frame to save memory
    def start_processing(self):
        self.running = True
        thread = threading.Thread(target=self._process_feed)
        thread.daemon = True
        thread.start()
        
    def _process_feed(self):
        try:
            self.cap = cv2.VideoCapture(self.config['url'])
            
            # Memory optimization settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 10)  # Limit FPS
            
            if not self.cap.isOpened():
                print(f"Failed to open video stream for {self.feed_id}")
                return
            frame_count = 0
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    # Video ended - restart
                    print(f"Video ended for {self.feed_id}, restarting...")
                    if self.cap:
                        self.cap.release()
                    
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.config['url'])
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if not self.cap.isOpened():
                        self.retry_count += 1
                        if self.retry_count >= self.max_retries:
                            print(f"Max retries reached for {self.feed_id}")
                            break
                        time.sleep(5)
                        continue
                    else:
                        self.retry_count = 0
                        frame_count = 0
                        continue
                # Skip frames to reduce processing load
                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue
                self.retry_count = 0
                # Resize frame immediately to save memory
                frame = cv2.resize(frame, (320, 240))
                # Detect persons with memory optimization
                try:
                    if model is not None:
                        results = model(frame, classes=[0], verbose=False, device='cpu')
                        detections = results[0].boxes
                        person_count = len(detections) if detections is not None else 0
                        
                        # Clear GPU memory if available
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        person_count = 0
                except Exception as e:
                    print(f"YOLO detection error for {self.feed_id}: {e}")
                    person_count = 0
                self.current_count = person_count
                # Log data less frequently
                if time.time() - self.last_logged >= self.log_interval:
                    self._log_data(person_count, frame)
                    self.last_logged = time.time()
                    
                    # Force garbage collection periodically
                    gc.collect()
                time.sleep(0.5)  # Longer delay to reduce CPU usage
        except Exception as e:
            print(f"Error processing {self.feed_id}: {e}")
        finally:
            if self.cap:
                self.cap.release()
            gc.collect()
    def _log_data(self, count, frame):
        timestamp = datetime.now()
        density_percentage = (count / self.config['max_capacity']) * 100
        # Determine alert level
        alert_level = "normal"
        if count >= alert_threshold:
            alert_level = "critical"
        elif count >= congestion_threshold:
            alert_level = "warning"
        # Store analytics data
        data_point = {
            "timestamp": timestamp.isoformat(),
            "count": count,
            "density_percentage": round(density_percentage, 2),
            "alert_level": alert_level,
            "location": self.config['location'],
            "area": self.config['area']
        }
        analytics_data[self.feed_id].append(data_point)
        # Update global feeds data
        feeds_data[self.feed_id] = {
            "name": self.config['name'],
            "current_count": count,
            "max_capacity": self.config['max_capacity'],
            "density_percentage": round(density_percentage, 2),
            "alert_level": alert_level,
            "last_updated": timestamp.isoformat(),
            "location": self.config['location'],
            "area": self.config['area']
        }
    def stop_processing(self):
        self.running = False
        if self.cap:
            self.cap.release()
            
# Initialize processors
feed_processors = {}
for feed_id, config in CCTV_FEEDS.items():
    feed_processors[feed_id] = FeedProcessor(feed_id, config)

    
# ------------------------- API ROUTES ---------------------------
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_feeds": len([f for f in feed_processors.values() if f.running])
    }

@app.get("/api/feeds")
async def get_all_feeds():
    return {
        "feeds": feeds_data,
        "total_count": sum(feed.get("current_count", 0) for feed in feeds_data.values()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/feeds/{feed_id}")
async def get_feed_details(feed_id: str):
    if feed_id not in feeds_data:
        return JSONResponse(status_code=404, content={"error": "Feed not found"})
    return {
        "feed": feeds_data[feed_id],
        "recent_analytics": list(analytics_data[feed_id])[-10:],
        "timestamp": datetime.now().isoformat()
    }

# Add these endpoints to your FastAPI app
@app.get("/api/video/stream/{feed_id}")
async def get_video_stream(feed_id: str):
    """Stream video for a specific feed"""
    if feed_id not in CCTV_FEEDS:
        return JSONResponse(status_code=404, content={"error": "Feed not found"})
    
    def generate_video_stream():
        cap = cv2.VideoCapture(CCTV_FEEDS[feed_id]['url'])
        
        # Set buffer size to avoid delays
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video reached for {feed_id}, restarting...")
                    # Restart video when it ends
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Failed to restart video for {feed_id}")
                        break
                
                # Resize frame for better streaming performance
                try:
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Encode frame as JPEG
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    ret_encode, buffer = cv2.imencode('.jpg', frame, encode_param)
                    
                    if not ret_encode:
                        print(f"Failed to encode frame for {feed_id}")
                        continue
                    
                    frame_bytes = buffer.tobytes()
                    
                    # Yield frame in multipart format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    # Control frame rate (adjust as needed)
                    time.sleep(0.033)  # ~30 FPS
                    
                except Exception as frame_error:
                    print(f"Frame processing error for {feed_id}: {frame_error}")
                    continue
                
        except Exception as e:
            print(f"Video stream error for {feed_id}: {e}")
        finally:
            if cap is not None:
                cap.release()
                print(f"Released video capture for {feed_id}")
    
    return StreamingResponse(
        generate_video_stream(), 
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )

@app.get("/api/video/snapshot/{feed_id}")
async def get_video_snapshot(feed_id: str):
    """Get current frame/snapshot from a feed"""
    if feed_id not in CCTV_FEEDS:
        return JSONResponse(status_code=404, content={"error": "Feed not found"})
    
    try:
        cap = cv2.VideoCapture(CCTV_FEEDS[feed_id]['url'])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return JSONResponse(status_code=500, content={"error": "Could not capture frame"})
        
        # Resize and encode frame
        frame = cv2.resize(frame, (320, 240))
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if not ret:
            return JSONResponse(status_code=500, content={"error": "Could not encode frame"})
        
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Snapshot error: {str(e)}"})
    
@app.get("/api/video/info")
async def get_video_info():
    """Get information about all video feeds"""
    video_info = {}
    
    for feed_id, config in CCTV_FEEDS.items():
        try:
            cap = cv2.VideoCapture(config['url'])
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                video_info[feed_id] = {
                    "name": config['name'],
                    "resolution": f"{width}x{height}",
                    "fps": fps,
                    "duration_seconds": frame_count / fps if fps > 0 else 0,
                    "total_frames": frame_count,
                    "status": "active"
                }
            else:
                video_info[feed_id] = {
                    "name": config['name'],
                    "status": "inactive",
                    "error": "Cannot open video source"
                }
            cap.release()
        except Exception as e:
            video_info[feed_id] = {
                "name": config['name'],
                "status": "error",
                "error": str(e)
            }
    
    return {"video_info": video_info}

@app.get("/api/analytics/summary")
async def get_analytics_summary():
    total_current = sum(feed.get("current_count", 0) for feed in feeds_data.values())
    total_capacity = sum(CCTV_FEEDS[feed_id]["max_capacity"] for feed_id in CCTV_FEEDS)
    alert_counts = {"critical": 0, "warning": 0, "normal": 0}
    for feed in feeds_data.values():
        alert_counts[feed.get("alert_level", "normal")] += 1
    one_hour_ago = datetime.now() - timedelta(hours=1)
    trend_data = []
    for feed_id, data_points in analytics_data.items():
        for point in data_points:
            if datetime.fromisoformat(point["timestamp"]) >= one_hour_ago:
                trend_data.append({
                    "feed_id": feed_id,
                    "feed_name": CCTV_FEEDS[feed_id]["name"],
                    **point
                })
    return {
        "summary": {
            "total_current_count": total_current,
            "total_capacity": total_capacity,
            "overall_density": round((total_current / total_capacity) * 100, 2) if total_capacity else 0,
            "alert_distribution": alert_counts,
            "active_feeds": len([f for f in feed_processors.values() if f.running])
        },
        "trend_data": trend_data,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/alerts")
async def get_current_alerts():
    alerts = [
        {
            "feed_id": fid,
            "feed_name": data["name"],
            "alert_level": data["alert_level"],
            "current_count": data["current_count"],
            "density_percentage": data["density_percentage"],
            "location": data["location"],
            "area": data["area"],
            "timestamp": data["last_updated"]
        }
        for fid, data in feeds_data.items() if data.get("alert_level") in ["warning", "critical"]
    ]
    return {"alerts": alerts, "count": len(alerts), "timestamp": datetime.now().isoformat()}

@app.get("/api/heatmap")
async def get_heatmap_data():
    heatmap_data = [
        {
            "location": data["location"],
            "intensity": data.get("density_percentage", 0),
            "count": data.get("current_count", 0),
            "area": data["area"],
            "name": data["name"],
            "alert_level": data.get("alert_level", "normal")
        }
        for data in feeds_data.values()
    ]
    return {"heatmap": heatmap_data, "timestamp": datetime.now().isoformat()}

@app.get("/api/predictions")
async def get_crowd_predictions():
    predictions = {}
    for feed_id, data_points in analytics_data.items():
        if len(data_points) >= 5:
            recent_counts = [point["count"] for point in list(data_points)[-10:]]
            avg_trend = np.mean(np.diff(recent_counts)) if len(recent_counts) > 1 else 0
            current = recent_counts[-1]
            pred_15 = max(0, int(current + avg_trend * 3))
            pred_30 = max(0, int(current + avg_trend * 6))
            risk = "high" if pred_15 >= alert_threshold else "medium" if pred_15 >= congestion_threshold else "low"
            predictions[feed_id] = {
                "feed_name": CCTV_FEEDS[feed_id]["name"],
                "current": current,
                "trend": round(avg_trend, 2),
                "predictions": {"15_min": pred_15, "30_min": pred_30},
                "risk_level": risk
            }
    return {"predictions": predictions, "timestamp": datetime.now().isoformat()}

@app.get("/api/export/csv")
async def export_data_csv():
    try:
        all_data = []
        for feed_id, data_points in analytics_data.items():
            for point in data_points:
                all_data.append({
                    "feed_id": feed_id,
                    "feed_name": CCTV_FEEDS[feed_id]["name"],
                    **point
                })
        df = pd.DataFrame(all_data)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return Response(content=buffer.read(), media_type="text/csv", headers={
            "Content-Disposition": f"attachment; filename=crowd_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/api/controls/start")
async def start_monitoring():
    try:
        for processor in feed_processors.values():
            if not processor.running:
                processor.start_processing()
        return {"message": "Monitoring started", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/api/controls/stop")
async def stop_monitoring():
    try:
        for processor in feed_processors.values():
            processor.stop_processing()
        return {"message": "Monitoring stopped", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get("/api/config/thresholds")
async def get_thresholds():
    return {
        "alert_threshold": alert_threshold,
        "congestion_threshold": congestion_threshold
    }

@app.post("/api/config/thresholds")
async def update_thresholds(body: Dict[str, int] = Body(...)):
    global alert_threshold, congestion_threshold
    if "alert_threshold" in body:
        alert_threshold = int(body["alert_threshold"])
    if "congestion_threshold" in body:
        congestion_threshold = int(body["congestion_threshold"])
    return {
        "message": "Thresholds updated",
        "alert_threshold": alert_threshold,
        "congestion_threshold": congestion_threshold
    }
@app.get("/api/stream/updates")
async def stream_updates():
    def event_stream():
        while True:
            data = {
                "feeds": feeds_data,
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(2)
    return StreamingResponse(event_stream(), media_type="text/event-stream")
# ------------------------- APP STARTUP ---------------------------
@app.on_event("startup")
async def startup_event():
    print("Starting Crowd Monitoring API (FastAPI)...")
    for processor in feed_processors.values():
        processor.start_processing()
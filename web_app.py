import cv2
from flask import Flask, render_template, Response, request
import face_recognition
import numpy as np
import pickle
from datetime import datetime, timedelta
import sqlite3 
import urllib3 
import os
import signal
import webbrowser 
import threading 

# Suppress warnings related to urllib3 when accessing network stream
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURATION ---
ENCODINGS_FILE = 'face_encodings.pkl'
DB_FILE_PATH = 'attendance_records.db'
ATTENDANCE_COOLDOWN_MINUTES = 1 # Cooldown is 1 minute for easy testing/demo.

# VITAL: CAMERA STREAM URL (Update this with the CURRENT IP from your mobile app)
CAMERA_URL = 'http://192.168.1.2:8080/video' 

# --- GLOBAL STATE ---
KNOWN_FACES = []
KNOWN_NAMES = []
last_attendance_time = {} 
SERVER_URL = "http://127.0.0.1:5000/"

# --- FLASK SETUP ---
app = Flask(__name__)

# ===============================================
# 1. DATABASE AND INITIALIZATION
# ===============================================

def init_db():
    """Initializes the SQLite database and creates the attendance table."""
    try:
        conn = sqlite3.connect(DB_FILE_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                log_time TEXT NOT NULL,
                log_date TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        print("INFO: Database initialized successfully.")
    except Exception as e:
        print(f"ERROR: Could not initialize database. {e}")

# Load Encodings
try:
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
        KNOWN_FACES = data["encodings"]
        KNOWN_NAMES = data["names"]
        print(f"Successfully loaded {len(KNOWN_FACES)} known face encodings.")
except FileNotFoundError:
    print("FATAL: Encodings file missing. Run train_model.py first!")

# Initialize Video Capture
def initialize_camera():
    print(f"INFO: Initializing camera via network stream: {CAMERA_URL}")
    video_capture = cv2.VideoCapture(CAMERA_URL) 
    
    if not video_capture.isOpened():
        print("FATAL ERROR: Could not open network stream. Check mobile app/IP.")
        return None

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return video_capture

camera = initialize_camera()

def log_attendance(face_name):
    """Logs the attendance to the SQLite database, respecting cooldown."""
    global last_attendance_time
    
    current_time = datetime.now()
    
    # Cooldown check
    if face_name in last_attendance_time:
        cooldown_expiry = last_attendance_time[face_name] + timedelta(minutes=ATTENDANCE_COOLDOWN_MINUTES)
        if current_time < cooldown_expiry:
            return
    
    try:
        conn = sqlite3.connect(DB_FILE_PATH)
        cursor = conn.cursor()
        
        timestamp = current_time.strftime("%H:%M:%S")
        log_date = current_time.strftime("%Y-%m-%d")

        cursor.execute(
            "INSERT INTO attendance (student_id, log_time, log_date) VALUES (?, ?, ?)",
            (face_name, timestamp, log_date)
        )
        conn.commit()
        conn.close()
            
        print(f"[{timestamp}] DB LOGGED: {face_name} on {log_date}.")
        last_attendance_time[face_name] = current_time
        
    except Exception as e:
        print(f"[ERROR] DB WRITE FAILED: {e}")

def get_attendance_by_day(date_str=None):
    """Retrieves attendance records for a specific date (YYYY-MM-DD)."""
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        
    cursor.execute(
        "SELECT student_id, log_time FROM attendance WHERE log_date = ? ORDER BY log_time DESC", 
        (date_str,)
    )
    records = cursor.fetchall()
    conn.close()
    
    output = f"--- Daily Attendance Report for {date_str} ---\n"
    if not records:
        output += "No records found for this date.\n"
    for student_id, log_time in records:
        output += f"{log_time}: {student_id} (PRESENT)\n"
    return output

def get_summary_report():
    """Retrieves a summary count of total logs per student (Analytics Feature)."""
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT student_id, COUNT(student_id) AS total_logs
        FROM attendance
        GROUP BY student_id
        ORDER BY total_logs DESC
    """)
    records = cursor.fetchall()
    conn.close()
    
    output = "--- Student Attendance Summary (Analytics) ---\n"
    if not records:
        output += "No attendance records available.\n"
    for student_id, total_logs in records:
        output += f"• {student_id}: {total_logs} Total Logs\n"
        
    return output

# ===============================================
# 2. STREAM GENERATOR (CORE LOGIC)
# ===============================================

def generate_frames():
    """Continuously processes frames and yields JPEG images for the web browser."""
    if camera is None:
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("WARNING: Stream read failed. Check mobile app status.")
            break
        
        # --- OPTIMIZED FACE RECOGNITION ---
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        status_text = "STATUS: Scan Face"
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            face_distances = face_recognition.face_distance(KNOWN_FACES, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = "Unknown"
            cooldown_active = False
            
            if face_distances[best_match_index] < 0.50: 
                name = KNOWN_NAMES[best_match_index]
                
                # Check cooldown status
                if name in last_attendance_time:
                    cooldown_expiry = last_attendance_time[name] + timedelta(minutes=ATTENDANCE_COOLDOWN_MINUTES)
                    if datetime.now() < cooldown_expiry:
                        cooldown_active = True

                # Log if not on cooldown
                if not cooldown_active:
                    log_attendance(name)
                    status_text = f"LOGGED: {name}"
                else:
                    status_text = f"COOLDOWN: {name}"

            # Scale coordinates back up for drawing on the full frame
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            color = (255, 0, 0) if cooldown_active or name == "Unknown" else (0, 255, 0)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        # Encode the frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ===============================================
# 3. FLASK ROUTES (Web Interface)
# ===============================================

@app.route('/')
def index():
    """Renders the main web page (Video Feed)."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint for the browser to receive the video stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs', methods=['GET', 'POST'])
def show_logs():
    """Displays attendance records for today or a selected date."""
    
    date_to_view = datetime.now().strftime("%Y-%m-%d")
    
    if request.method == 'POST':
        date_to_view = request.form.get('view_date')
        if not date_to_view:
             date_to_view = datetime.now().strftime("%Y-%m-%d")
    
    log_content = get_attendance_by_day(date_to_view)
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    return render_template(
        'logs.html', 
        log_content=log_content,
        today_date=today_date,
        date_to_view=date_to_view
    )
    
@app.route('/summary')
def show_summary():
    """Renders a page showing the overall attendance summary."""
    summary_content = get_summary_report()
    
    return render_template(
        'logs.html', 
        log_content=summary_content,
        today_date="",
        date_to_view="Overall Summary Report"
    )

@app.route('/shutdown', methods=['GET'])
def shutdown_app():
    """Shuts down the Flask server gracefully via web command."""
    # Send signal to stop the main process 
    os.kill(os.getpid(), signal.SIGINT)
    
    print("\nINFO: Application shutting down by web command...")
    return "Application shutting down..."

def open_browser():
    """Function to open the browser automatically after a short delay."""
    # Timer ensures Flask server is fully initialized before the browser attempts connection.
    threading.Timer(1.5, lambda: webbrowser.open(SERVER_URL)).start()


if __name__ == '__main__':
    init_db() 
    open_browser() 
    print("--- Web Attendance Server Starting ---")
    print(f"Access the application at: {SERVER_URL}")
    
    # Run the server on all available interfaces (0.0.0.0)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

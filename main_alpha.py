import cv2
import os
import numpy as np
#from ultralogger import Logger
from ultralytics import YOLO
from error_script import VIDEO_ERRORS

# --- 1. CONFIGURATION ---
MODEL_PATH = 'training_runs/assembly_obb_v1/weights/best.pt'
VIDEO_PATH = 'bis.mp4'
OUTPUT_FOLDER = 'output'
OUTPUT_VIDEO_NAME = 'dashboard_demo.mp4'
CONFIDENCE_THRESHOLD = 0.4

# --- Dashboard Layout ---
# The video will be resized to this, so a 16:9 ratio is best (e.g., 1280x720)
VIDEO_WIDTH = 854
VIDEO_HEIGHT = 480
LOG_PANEL_WIDTH = 500 # Width of the log panel on the right

# --- Colors for the Log ---
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)

# Your class names from the data.yaml file
CLASS_NAMES = [
    'bolt', 'bolt misplaced', 'door', 'door-panel', 'door-panel fitting',
    'objects', 'panel gap', 'panel gap misallignment', 'sealant-gun',
    'Sealent Issue', 'tyre', 'tyre damage', 'window-glass', 'windshield',
    'worker', 'wrench'
]

# --- 2. DEFINE THE "GOLDEN STANDARD" PROCESS ---
# IMPORTANT: You must edit the 'trigger_frame' to match the timing of your video!
# This tells the log when to mark a step as "OK".
GOLDEN_STANDARD_STEPS = [
    {"step": "1. Door Panel Alignment", "status": "PENDING", "trigger_frame": 100},
    {"step": "2. Secure Mounting Bolts", "status": "PENDING", "trigger_frame": 250},
    {"step": "3. Connect Wiring Harness", "status": "PENDING", "trigger_frame": 400},
    {"step": "4. Final Panel Seating", "status": "PENDING", "trigger_frame": 550},
    {"step": "5. Quality Check Complete", "status": "PENDING", "trigger_frame": 700}
]

# --- 3. HELPER FUNCTION TO CREATE THE LOG PANEL ---
def create_log_panel(height, width, steps, active_error):
    """Creates the black panel on the right with status text."""
    log_panel = np.zeros((height, width, 3), dtype=np.uint8)
    
    # --- Title ---
    cv2.putText(log_panel, "PROCESS LOG", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 3)
    cv2.line(log_panel, (30, 70), (width - 30, 70), COLOR_WHITE, 1)

    # --- Golden Standard Steps ---
    y_offset = 120
    for item in steps:
        status = item["status"]
        text = item["step"]
        
        if status == "PENDING":
            color = COLOR_YELLOW
            display_text = f"[?] {text}"
        elif status == "OK":
            color = COLOR_GREEN
            display_text = f"[\u2713] {text}" # Unicode checkmark
        else: # FAILED
            color = COLOR_RED
            display_text = f"[X] {text}"

        cv2.putText(log_panel, display_text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        y_offset += 40

    # --- Anomaly Description ---
    if active_error:
        cv2.putText(log_panel, "CURRENT ANOMALY:", (30, height - 150), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_YELLOW, 2)
        
        # Wrap text for the description
        line1 = active_error["message"]
        desc = "Process halted. Manual check required."
        
        cv2.putText(log_panel, line1, (30, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_RED, 2)
        cv2.putText(log_panel, desc, (30, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
        
    return log_panel

# --- 4. MAIN FUNCTION ---
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_video_path = os.path.join(OUTPUT_FOLDER, OUTPUT_VIDEO_NAME)

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file at '{VIDEO_PATH}'")
        return

    # Video writer for the final dashboard output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (VIDEO_WIDTH + LOG_PANEL_WIDTH, VIDEO_HEIGHT))

    frame_count = 0

    print("Processing video to create dashboard... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Resize frame to fit the left panel
        video_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

        # Run detection on the resized frame
        results = model(video_frame, stream=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
        if results:
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(video_frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)

        # Check for pre-scripted errors
        active_error = None
        for error in VIDEO_ERRORS:
            if error["start_frame"] <= frame_count <= error["end_frame"]:
                active_error = error
                # Highlight error on the video frame
                roi = error["roi"]
                cv2.rectangle(video_frame, (roi[0], roi[1]), (roi[2], roi[3]), COLOR_RED, 4)
                cv2.putText(video_frame, error["message"], (roi[0], roi[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_RED, 3)
                break

        # Update the status of the golden standard steps based on frame count
        for step in GOLDEN_STANDARD_STEPS:
            if frame_count >= step["trigger_frame"]:
                step["status"] = "OK"

        # Create the log panel
        log_panel = create_log_panel(VIDEO_HEIGHT, LOG_PANEL_WIDTH, GOLDEN_STANDARD_STEPS, active_error)

        # Combine the video frame and the log panel into one screen
        dashboard_screen = np.concatenate((video_frame, log_panel), axis=1)

        # Show the final dashboard and save it
        cv2.imshow("VLLM Assembly Inspector Dashboard", dashboard_screen)
        out.write(dashboard_screen)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"\nDashboard video saved to: {output_video_path}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# --- 5. SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    main()
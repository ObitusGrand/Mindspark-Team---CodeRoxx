import cv2
import os
import numpy as np
from ultralytics import YOLO
from error_script import VIDEO_ERRORS
from PIL import Image, ImageDraw, ImageFont

# ====================================================================================
# --- 1. CONFIGURATION ---
# ====================================================================================

# --- File Paths ---
MODEL_PATH = 'training_runs/assembly_obb_v1/weights/best.pt'
VIDEO_PATH = 'bis.mp4'
FONT_PATH = 'orbitron.ttf'

# --- Output Settings ---
OUTPUT_FOLDER = 'output'
OUTPUT_VIDEO_NAME = 'final_dashboard_854x480.mp4'

# --- Detection Settings ---
CONFIDENCE_THRESHOLD = 0.4

# --- Class Names ---
CLASS_NAMES = [
    'bolt', 'bolt misplaced', 'door', 'door-panel', 'door-panel fitting',
    'objects', 'panel gap', 'panel gap misallignment', 'sealant-gun',
    'Sealent Issue', 'tyre', 'tyre damage', 'window-glass', 'windshield',
    'worker', 'wrench'
]

# --- Golden Standard Steps ---
GOLDEN_STANDARD_STEPS = [
    {"step": "Sealant Fix", "status": "PENDING", "trigger_frame": 100},
    {"step": "Grabbing Windshield", "status": "PENDING", "trigger_frame": 250},
    {"step": "Windshield Sealant", "status": "PENDING", "trigger_frame": 400},
    {"step": "Winshield Seating", "status": "PENDING", "trigger_frame": 550},
    {"step": "Quality Check Complete", "status": "PENDING", "trigger_frame": 700}
]

# ====================================================================================
# --- 2. UI THEME AND LAYOUT ---
# ====================================================================================
# Using your specified resolution
VIDEO_WIDTH, VIDEO_HEIGHT = 854, 480
LOG_PANEL_WIDTH = 550

COLORS = {
    "bg_dark": (33, 39, 51), "panel_bg": (48, 56, 70), "text_light": (224, 224, 224),
    "accent_cyan": (0, 255, 255), "status_ok": (57, 255, 20),
    "status_pending": (255, 191, 0), "status_fail": (255, 49, 49)
}

try:
    # Font sizes are slightly reduced to better fit the smaller window
    FONT_TITLE = ImageFont.truetype(FONT_PATH, 36)
    FONT_BODY = ImageFont.truetype(FONT_PATH, 22)
    FONT_SMALL = ImageFont.truetype(FONT_PATH, 18)
except IOError:
    print(f"FATAL ERROR: The font file was not found at '{FONT_PATH}'.")
    exit()

# ====================================================================================
# --- 3. CORE APPLICATION LOGIC ---
# ====================================================================================

def create_log_panel(height, width, steps, active_error_msg=None):
    """Creates the aesthetic log panel with dynamic layout."""
    panel_pil = Image.new('RGB', (width, height), COLORS["bg_dark"])
    draw = ImageDraw.Draw(panel_pil)
    
    draw.text((40, 30), "SYSTEM LOG", font=FONT_TITLE, fill=COLORS["accent_cyan"])
    draw.line([(40, 80), (width - 40, 80)], fill=COLORS["panel_bg"], width=2)
    
    y_offset = 110
    draw.text((40, y_offset), "PROCESS STEPS", font=FONT_BODY, fill=COLORS["text_light"])
    y_offset += 35
    for item in steps:
        status, text = item["status"], item["step"]
        color = COLORS.get(f"status_{status.lower()}", COLORS["text_light"])
        symbol = {"OK": "[✓]", "PENDING": "[~]", "FAILED": "[✗]"}.get(status, "[?]")
        draw.text((50, y_offset), f"{symbol} {text}", font=FONT_SMALL, fill=color)
        y_offset += 30

    # Dynamic positioning ensures this section is always visible
    anomaly_y_start = y_offset + 40
    if anomaly_y_start > height - 120:
        anomaly_y_start = height - 120
        
    draw.line([(40, anomaly_y_start), (width - 40, anomaly_y_start)], fill=COLORS["panel_bg"], width=2)
    anomaly_y_start += 15
    draw.text((40, anomaly_y_start), "LIVE ANOMALIES", font=FONT_BODY, fill=COLORS["text_light"])
    
    anomaly_y_start += 40
    if active_error_msg:
        draw.text((50, anomaly_y_start), f"TYPE: {active_error_msg}", font=FONT_SMALL, fill=COLORS["status_fail"])
        draw.text((50, anomaly_y_start + 30), "ACTION: Halt & Inspect", font=FONT_SMALL, fill=COLORS["text_light"])
    else:
        draw.text((50, anomaly_y_start), "STATUS: Nominal", font=FONT_SMALL, fill=COLORS["status_ok"])

    return cv2.cvtColor(np.array(panel_pil), cv2.COLOR_RGB2BGR)

def main():
    """Main function to run the assembly inspection dashboard."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_video_path = os.path.join(OUTPUT_FOLDER, OUTPUT_VIDEO_NAME)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"FATAL ERROR: Could not open video file at '{VIDEO_PATH}'.")
        return
    
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (VIDEO_WIDTH + LOG_PANEL_WIDTH, VIDEO_HEIGHT))
    frame_count = 0

    print("Processing video... Press 'q' in the display window to quit early.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video processing complete.")
            break
        
        frame_count += 1
        video_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

        # --- Anomaly and Step Logic ---
        active_error_message = None
        for error in VIDEO_ERRORS:
            if error["start_frame"] <= frame_count <= error["end_frame"]:
                active_error_message = error['message']
                roi = error["roi"]
                # Draw anomaly highlight on video
                cv2.rectangle(video_frame, (roi[0], roi[1]), (roi[2], roi[3]), COLORS["status_fail"], 4)
                cv2.putText(video_frame, "!", (roi[0], roi[1] - 10), cv2.FONT_HERSHEY_TRIPLEX, 1.5, COLORS["status_fail"], 3)
                break
        
        for step in GOLDEN_STANDARD_STEPS:
            if frame_count >= step["trigger_frame"]:
                step["status"] = "OK"

        # --- Build and Display Dashboard ---
        log_panel = create_log_panel(VIDEO_HEIGHT, LOG_PANEL_WIDTH, GOLDEN_STANDARD_STEPS, active_error_message)
        dashboard_screen = np.concatenate((video_frame, log_panel), axis=1)
        cv2.imshow("VLLM Assembly Inspector Dashboard", dashboard_screen)
        out.write(dashboard_screen)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"\nDashboard video saved successfully to: {output_video_path}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
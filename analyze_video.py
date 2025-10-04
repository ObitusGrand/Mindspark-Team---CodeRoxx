import cv2
from error_script import VIDEO_ERRORS

# --- CONFIGURATION ---
# This path must match the video you are using in main.py
VIDEO_PATH = 'bis.mp4' 

# --- SCRIPT ---
print("\n--- Video and Error Script Analyzer ---")

# 1. Analyze the Video File
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ FATAL ERROR: Could not open video file at '{VIDEO_PATH}'.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration_seconds = total_frames / fps

print(f"\n✅ Video Analysis for '{VIDEO_PATH}':")
print(f"   - Total Frames: {total_frames}")
print(f"   - Duration: {duration_seconds:.2f} seconds")
print("----------------------------------------")

# 2. Analyze the error_script.py File
print("\n✅ Analyzing `VIDEO_ERRORS` from your script:")

if not VIDEO_ERRORS:
    print("   - ❌ WARNING: The `VIDEO_ERRORS` list is empty! No anomalies can be shown.")
else:
    for i, error in enumerate(VIDEO_ERRORS):
        start = error.get("start_frame", "N/A")
        end = error.get("end_frame", "N/A")
        msg = error.get("message", "No message")
        
        print(f"\n   --- Error #{i+1} ---")
        print(f"   - Message: '{msg}'")
        print(f"   - Starts at frame: {start}")
        print(f"   - Ends at frame: {end}")
        
        # 3. Compare and Report
        if isinstance(start, int) and start > total_frames:
            print(f"   - ❌ CRITICAL ISSUE: This error will NEVER trigger.")
            print(f"   - REASON: The error starts at frame {start}, but the video only has {total_frames} frames.")
        elif isinstance(start, int):
            start_time = start / fps
            print(f"   - ✅ This error should trigger at approximately {start_time:.2f} seconds into the video.")
        else:
            print("   - ❌ WARNING: 'start_frame' key is missing or invalid for this error.")

print("\n----------------------------------------")
print("Analysis complete.\n")

cap.release()
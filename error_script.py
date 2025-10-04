# error_script.py

# This is the "script" for your demo.
# Define all known errors in the video you want to demonstrate.
#
# How to fill this out:
# 1. Watch your video frame by frame.
# 2. When an error starts, note the "start_frame".
# 3. When the error ends, note the "end_frame".
# 4. Use an image editor (like MS Paint) to find the pixel coordinates [x1, y1, x2, y2]
#    of a box around the error.
# 5. Write a clear "message" for the judges to read.

VIDEO_ERRORS = [
    # --- EXAMPLE 1: A panel gap ---
    {
        "start_frame": 250,
        "end_frame": 400,
        "message": "ANOMALY: Panel Gap Misalignment!",
        "roi": [810, 350, 980, 550]  # [top-left-x, top-left-y, bottom-right-x, bottom-right-y]
    },

    # --- EXAMPLE 2: A misplaced bolt ---
    {
        "start_frame": 520,
        "end_frame": 650,
        "message": "WARNING: Bolt Misplaced!",
        "roi": [400, 600, 550, 720]
    },

    # --- EXAMPLE 3: A sealant issue on a windshield ---
    {
        "start_frame": 700,
        "end_frame": 850,
        "message": "CRITICAL: Sealant Application Error!",
        "roi": [200, 250, 350, 450]
    }

    # Add more error dictionaries here for every anomaly in your video.
]
# debug_detections.py
from ultralytics import YOLO
import cv2

# --- CONFIGURATION ---
# Make sure these paths are correct
MODEL_PATH = 'training_runs/assembly_obb_v1/weights/best.pt'
IMAGE_PATH = 'assets/image.jpg'

# The list of class names your model was trained on
CLASS_NAMES = [
    'bolt', 'bolt misplaced', 'door', 'door-panel', 'door-panel fitting',
    'objects', 'panel gap', 'panel gap misallignment', 'sealant-gun',
    'Sealent Issue', 'tyre', 'tyre damage', 'window-glass', 'windshield',
    'worker', 'wrench'
]

# --- SCRIPT ---
print("--- Starting Model Debugger ---")

# Load the model
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Run inference on the image
# We set conf=0.01 to see EVERYTHING the model detects, no matter how low the confidence
try:
    results = model(IMAGE_PATH, conf=0.01) 
    print(f"✅ Model inference complete.")
except Exception as e:
    print(f"❌ Error during model inference: {e}")
    exit()

# The most important part: Print the raw results
print("\n--- RAW MODEL OUTPUT ---")
detections_found = False
for r in results:
    if r.boxes:
        detections_found = True
        print(r.boxes) # This prints all detected boxes with their confidences

if not detections_found:
    print("❌ No objects were detected in the image.")
    print("--------------------------")
    exit()

print("--------------------------\n")


# --- VISUALIZATION ---
# Draw the boxes on the image so you can see them
img = cv2.imread(IMAGE_PATH)
for r in results:
    if r.boxes:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id]

            # Draw box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

print("✅ Displaying image with detected boxes. Press any key to exit.")
cv2.imshow("Debug Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
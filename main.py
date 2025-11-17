import cv2
import mediapipe as mp
import numpy as np
import os

# ========== SETTINGS ==========
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe FaceMesh initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=20,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Face oval indices (MediaPipe official)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

def blur_face_oval(frame, landmarks):
    h, w, _ = frame.shape
    pts = []

    # Extract face oval coordinates
    for idx in FACE_OVAL:
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append([x, y])

    pts = np.array(pts)

    if len(pts) < 5:
        return frame

    # Fit ellipse to oval points
    ellipse = cv2.fitEllipse(pts)

    # Mask for blurring
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)

    # Strong blur
    blurred = cv2.GaussianBlur(frame, (75, 75), 50)

    # Merge blurred & original
    return np.where(mask[..., None] == 255, blurred, frame)

# ========== PROCESS VIDEO ==========

def process_video(input_path):
    if not os.path.exists(input_path):
        print("❌ Video file not found.")
        return

    video_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"{OUTPUT_DIR}/{video_name}_blurred.mp4"

    cap = cv2.VideoCapture(input_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🚀 Processing: {input_path}")
    print(f"💾 Saving to: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame = blur_face_oval(frame, face_landmarks)

        out.write(frame)

    cap.release()
    out.release()

    print("✅ Done! Saved blurred video.")

# ========== RUN ==========

if __name__ == "__main__":
    video_path = input("Enter video path: ").strip()
    process_video(video_path)

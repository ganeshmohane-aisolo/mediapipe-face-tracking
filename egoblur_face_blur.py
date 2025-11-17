import os
import cv2
import numpy as np
import torch

# ================================================================
#                 Load EgoBlur Face Model (.jit)
# ================================================================
MODEL_PATH = "models/ego_blur_face.jit"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()

# ================================================================
#                 Natural Ellipse Blur Function
# ================================================================
def natural_ellipse_blur(frame, x1, y1, x2, y2):
    H, W = frame.shape[:2]

    # Clip within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W - 1, x2)
    y2 = min(H - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return frame

    crop = frame[y1:y2, x1:x2]
    blur = cv2.GaussianBlur(crop, (75, 75), 40)

    h, w = crop.shape[:2]
    center = (w // 2, h // 2)
    axes = (int(w * 0.45), int(h * 0.60))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Feather edges
    mask = cv2.GaussianBlur(mask, (51, 51), 30)
    mask_f = (mask / 255.0)[..., None]

    blended = (crop * (1 - mask_f) + blur * mask_f).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended

    return frame


# ================================================================
#             Run EgoBlur Detection On a Single Frame
# ================================================================
def detect_faces_egoblur(frame):
    """
    Converts frame → tensor → model → bounding boxes.
    Output expected: Tensor of shape [N, 4] containing (x1,y1,x2,y2)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    img = img.unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        out = model(img)

    # Convert tensor to numpy
    if isinstance(out, (tuple, list)):
        boxes = out[0]
    else:
        boxes = out

    boxes = boxes.cpu().numpy().astype(int)
    return boxes


# ================================================================
#                     Video Processing Pipeline
# ================================================================
def process_video(input_path, output_folder="outputs"):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "egoblur_output.mp4")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open input video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    print("🚀 EgoBlur processing started...")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # ---- Run EgoBlur Model ----
        boxes = detect_faces_egoblur(frame)

        # ---- Apply Blur ----
        for (x1, y1, x2, y2) in boxes:
            frame = natural_ellipse_blur(frame, x1, y1, x2, y2)

        writer.write(frame)

        if frame_id % 20 == 0:
            print(f"Processed {frame_id} frames...")

    cap.release()
    writer.release()

    print(f"✅ Completed! Saved at: {output_path}")


# ================================================================
#                       MAIN ENTRY POINT
# ================================================================
if __name__ == "__main__":
    input_video = "inputs/crowd.mp4"  # <- change your input file here
    process_video(input_video)

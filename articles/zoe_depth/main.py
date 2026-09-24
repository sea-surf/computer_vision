import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTS = (".mp4",)

# Global paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(ROOT_DIR, "video.mp4")
OUTPUT_DIR = os.path.join(ROOT_DIR, "predict")
MODEL_PATH = os.path.join(ROOT_DIR, "best.pt")


def process_frame(frame, yolo_model, depth_estimator, width, height):
    """
    Runs YOLO fire detection + ZoeDepth metric depth on a single frame/image.
    Draws overlays directly onto `frame` (modified in place) and returns it,
    along with the distance to the biggest fire box (or None if no fire found).
    """
    # YOLO inference - collect fire boxes only
    results = yolo_model(frame, verbose=False)
    fire_boxes = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = yolo_model.names[cls_id]
            if class_name.lower() != 'fire':
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Clamp to frame bounds to avoid empty/invalid slices
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))

            if x2 > x1 and y2 > y1:
                fire_boxes.append((x1, y1, x2, y2))

    best_distance = None

    # Only run ZoeDepth if there's actually fire in this frame
    if fire_boxes:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)  # converts cv frame to rgb frame for the model
        depth_result = depth_estimator(pil_frame)  # runs the depth estimator model

        full_depth_tensor = depth_result["predicted_depth"]

        # --- RAW, explicit tensor handling (no blind .squeeze()) ---
        # Convert to numpy first, keeping whatever shape it came in as.
        if torch.is_tensor(full_depth_tensor):
            depth_tensor_np = full_depth_tensor.detach().cpu().numpy()
        else:
            depth_tensor_np = np.array(full_depth_tensor)

        print(f"[DEBUG] raw predicted_depth shape: {depth_tensor_np.shape}")

        # Explicitly strip leading dimensions (e.g. batch, or extra head dim)
        # one at a time, always taking index 0, until we're left with a
        # plain 2D (H, W) depth map. This is deterministic and won't
        # accidentally collapse a real spatial dimension the way squeeze() can.
        while depth_tensor_np.ndim > 2:
            depth_tensor_np = depth_tensor_np[0]

        print(f"[DEBUG] depth map after reducing to 2D: {depth_tensor_np.shape}, "
              f"min={depth_tensor_np.min():.3f}, max={depth_tensor_np.max():.3f}")

        if depth_tensor_np.shape[:2] != (height, width):
            depth_tensor_np = cv2.resize(
                depth_tensor_np, (width, height), interpolation=cv2.INTER_NEAREST
            )

        # Global normalization (frame-wide) so colors are comparable across boxes
        depth_min = float(depth_tensor_np.min())
        depth_max = float(depth_tensor_np.max())
        depth_range = depth_max - depth_min
        if depth_range > 1e-6:
            depth_vis_full = ((depth_tensor_np - depth_min) / depth_range * 255).astype(np.uint8)
        else:
            depth_vis_full = np.zeros_like(depth_tensor_np, dtype=np.uint8)

        biggest_area = 0

        for (x1, y1, x2, y2) in fire_boxes:
            w, h = x2 - x1, y2 - y1

            # Crop metric depth values for the bounding box
            box_depth_tensor = depth_tensor_np[y1:y2, x1:x2]  # crop just the fire box region
            mean_depth = float(np.mean(box_depth_tensor))  # average all pixels in that box into ONE number

            if w * h > biggest_area:
                biggest_area = w * h
                best_distance = mean_depth

            # Crop the pre-normalized (frame-wide) depth visualization
            depth_vis = depth_vis_full[y1:y2, x1:x2]
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

            # Overlay depth map back onto the frame crop
            crop = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = cv2.addWeighted(crop, 0.3, depth_colormap, 0.7, 0)

            # Draw bounding box + per-box distance label (blue for contrast against fire/orange tones)
            box_color = (255, 100, 0)  # BGR: blue
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame, f"Fire: {mean_depth:.2f}m", (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2
            )

        # Display best (biggest fire) distance on top right corner
        if best_distance is not None:
            text = f"Biggest Fire Dist: {best_distance:.2f}m"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = width - text_size[0] - 20
            text_y = 40
            cv2.putText(
                frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
            )

    return frame, best_distance


def process_images(input_dir, output_dir, yolo_model, depth_estimator):
    """
    Processes every image file (IMAGE_EXTS) found directly inside input_dir,
    running fire detection + depth overlay on each, and saving results to output_dir.
    """
    for file_name in os.listdir(input_dir):
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in IMAGE_EXTS:
            continue

        image_path = os.path.join(input_dir, file_name)
        out_path = os.path.join(output_dir, file_name)

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to open {image_path}")
            continue

        height, width = frame.shape[:2]
        frame, _ = process_frame(frame, yolo_model, depth_estimator, width, height)

        cv2.imwrite(out_path, frame)
        print(f"Processed {file_name}")


def process_videos(input_dir, output_dir, yolo_model, depth_estimator):
    """
    Processes every video file (VIDEO_EXTS) found directly inside input_dir,
    running fire detection + depth overlay on each frame, and saving results to output_dir.
    """
    for file_name in os.listdir(input_dir):
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in VIDEO_EXTS:
            continue

        video_path = os.path.join(input_dir, file_name)
        out_path = os.path.join(output_dir, file_name)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, _ = process_frame(frame, yolo_model, depth_estimator, width, height)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"Processed {file_name}")


def run():
    """
    Loads models once, then processes every image and every video
    found in INPUT_DIR, saving all results to OUTPUT_DIR.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load YOLO
    yolo_model = YOLO(MODEL_PATH)

    # Load ZoeDepth pipeline
    device = 0 if torch.cuda.is_available() else -1
    depth_estimator = pipeline(
        task="depth-estimation",
        model="Intel/zoedepth-nyu-kitti",  # outdoors+indoors (nyu-kitti) / outdoors (kitti)
        device=device
    )

    process_images(INPUT_DIR, OUTPUT_DIR, yolo_model, depth_estimator)
    process_videos(INPUT_DIR, OUTPUT_DIR, yolo_model, depth_estimator)


if __name__ == "__main__":
    run()
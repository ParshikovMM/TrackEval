import os
import shutil
from pathlib import Path

import cv2
import pandas as pd

# === Параметры ===
AICITY_DIRS = ["/mnt/data/TrackEval/data/gt/mot_challenge/AICity22/train",
               "/mnt/data/TrackEval/data/gt/mot_challenge/AICity22/validation"]
OUTPUT_DIR = "/mnt/data/TrackEval/data/gt/mot_challenge/AICity22/all"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_video_info(video_path):
    """Извлечь FPS, разрешение и количество кадров из видео."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Не удалось открыть видео: {video_path}")
        return 30, 1920, 1080
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height


def create_seqinfo(scene, cam, gt_path, output_dir):
    df = pd.read_csv(gt_path, header=None)

    fps, width, height = get_video_info(Path(gt_path.parents[1], "vdo.avi"))

    max_frame = int(df[0].max())
    seq_name = f"{scene}_{cam}"

    seqinfo_content = f"""[Sequence]
name={seq_name}
imDir=.
frameRate={fps}
seqLength={max_frame}
imWidth={width}
imHeight={height}
imExt=.jpg
"""
    seqinfo_path = os.path.join(output_dir, "seqinfo.ini")
    with open(seqinfo_path, "w") as f:
        f.write(seqinfo_content)


def process_aicity_dir(root_dir):
    for scene in sorted(os.listdir(root_dir)):
        scene_path = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        for cam in sorted(os.listdir(scene_path)):
            cam_path = os.path.join(scene_path, cam)
            if not os.path.isdir(cam_path):
                continue

            gt_path = Path(cam_path, "gt", "gt.txt")

            if os.path.exists(gt_path):
                out_file = Path(OUTPUT_DIR, f"{scene}_{cam}", "gt", "gt.txt")
                out_file.parent.mkdir(parents=True, exist_ok=True)

                create_seqinfo(scene, cam, gt_path, out_file.parents[1])

                shutil.copy2(gt_path, out_file)
            else:
                print(f"Не найден gt для {scene}/{cam}")


if __name__ == "__main__":
    for base_dir in AICITY_DIRS:
        process_aicity_dir(base_dir)

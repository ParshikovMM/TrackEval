import os
import shutil
from pathlib import Path

TRACKER_DIR = "/mnt/data/TrackEval/data/trackers/mot_challenge/mipt-train/CarSORT-botkalman-stopping_raw"  # корневая папка с результатами трекера
SOURCE_FILENAME = "data.txt"  # имя файла с результатами (из которого брать данные)
TARGET_EXTENSION = ".txt"  # формат, ожидаемый TrackEval
TARGET_DIR = "/mnt/data/TrackEval/data/trackers/mot_challenge/mipt-train/CarSORT-botkalman-stopping/data"


def flatten_tracker_results(tracker_dir):
    for sub in sorted(os.listdir(tracker_dir)):
        sub_path = os.path.join(tracker_dir, sub)
        if not os.path.isdir(sub_path):
            continue

        src_file = os.path.join(sub_path, SOURCE_FILENAME)
        if not os.path.exists(src_file):
            print(f"⚠️ Пропущено {sub} — нет {SOURCE_FILENAME}")
            continue

        # dst_file = os.path.join(tracker_dir, f"{sub}{TARGET_EXTENSION}")
        dst_file = Path(TARGET_DIR, f"{sub}{TARGET_EXTENSION}")

        # Копируем содержимое
        shutil.copy2(src_file, dst_file)
        print(f"✅ {sub}: {SOURCE_FILENAME} → {os.path.basename(dst_file)}")

    print("\n🎯 Готово: результаты преобразованы в MOTChallenge-совместимый формат.")


if __name__ == "__main__":
    flatten_tracker_results(TRACKER_DIR)

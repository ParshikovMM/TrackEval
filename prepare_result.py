import pandas as pd

# Путь к твоему файлу с результатами (например: results/mot_challenge/AICity22-train_BotSORT.csv)
INPUT_CSV = "/mnt/data/TrackEval/data/trackers/mot_challenge/mipt-train/CarSORT-botkalman-stopping/pedestrian_detailed.csv"
OUTPUT_CSV = "/mnt/data/TrackEval/data/trackers/mot_challenge/mipt-train/CarSORT-botkalman-stopping/pedestrian_detailed_summary.csv"

# Загружаем все метрики
df = pd.read_csv(INPUT_CSV)

# Проверим, какие колонки есть
print("Колонки в исходном CSV:", df.columns.tolist()[:20], "...")

# Выбираем только нужные
keep_cols = [
    "seq",
    "HOTA(0)",
    "DetA___AUC",
    "AssA___AUC",
    "MOTA",
    "IDF1",
    "CLR_FP",   # FP
    "CLR_FN",   # FN
    "IDSW"      # ID switches
]

# Фильтруем только эти колонки, если они есть в CSV
keep_cols = [c for c in keep_cols if c in df.columns]
summary = df[keep_cols].copy()

# Добавим строку COMBINED (если есть) в конец таблицы
if "COMBINED" in summary["seq"].values:
    combined = summary[summary["seq"] == "COMBINED"]
    summary = pd.concat([summary[summary["seq"] != "COMBINED"], combined])

# Сохраняем таблицу
summary.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Сводная таблица сохранена в {OUTPUT_CSV}")

print("\n📊 Итог:")
print(summary.to_string(index=False))

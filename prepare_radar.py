import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Путь к файлу ===
CSV_PATH = "/mnt/data/TrackEval/data/trackers/mot_challenge/mipt-train/summary.csv"

# === Метрики и их лимиты (min, max) ===
METRICS_LIMITS = {
    "HOTA": (0.6, 1.0),
    "MOTA": (0.6, 1.0),
    "IDF1": (0.6, 1.0),
    "DetA": (0.6, 1.0),
    "AssA": (0.6, 1.0),
}

# === Загружаем данные ===
df = pd.read_csv(CSV_PATH)

# Проверяем наличие колонок
for m in METRICS_LIMITS.keys():
    if m not in df.columns:
        raise ValueError(f"Не найдена метрика {m} в таблице")

trackers = df["tracker"].tolist()
metrics = list(METRICS_LIMITS.keys())
values = df[metrics].to_numpy()

# === Нормируем значения под лимиты (чтобы корректно отрисовать)
norm_values = []
for i in range(len(values)):
    normalized = []
    for j, m in enumerate(metrics):
        min_v, max_v = METRICS_LIMITS[m]
        val = (values[i][j] - min_v) / (max_v - min_v)
        normalized.append(np.clip(val, 0, 1))  # за пределы не выходить
    norm_values.append(normalized)
values = np.array(norm_values)

# === Подготовка углов для "розы ветров"
num_vars = len(metrics)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # замыкаем круг

# === Фигура ===
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# === Отрисовка ===
for i, tracker in enumerate(trackers):
    vals = values[i].tolist()
    vals += vals[:1]  # замыкаем круг
    ax.plot(angles, vals, linewidth=2, label=tracker)
    ax.fill(angles, vals, alpha=0.25)

# === Настройка осей ===
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]))

# === Подписи метрик с min/max ===
labels_with_limits = []
for m in metrics:
    min_v, max_v = METRICS_LIMITS[m]
    labels_with_limits.append(f"{m}\n({min_v:.2f}–{max_v:.2f})")
ax.set_thetagrids(np.degrees(angles[:-1]), labels_with_limits, fontsize=11)

# === Радиальная шкала ===
ax.set_rlabel_position(0)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["", "", "", ""])  # убираем подписи радиусов, чтобы не мешали

# === Легенда ===
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
plt.title("Сравнение трекеров на MIPT", fontsize=15, pad=20)
plt.tight_layout()
plt.show()

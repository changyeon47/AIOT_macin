import pandas as pd
import matplotlib.pyplot as plt

# 1) 비교할 csv들 경로 & 라벨 설정
csv_infos = [
    {
        "path": r"D:\dataset\AIhub\runs\detect\orgin4\results.csv",
        "label": "YOLOv8n"
    },
    {
        "path": r"D:\dataset\AIhub\runs\detect\fine_tune_ft5\results.csv",
        "label": "fine_tune(epochs=100)"
    },
    {
        "path": r"D:\dataset\AIhub\runs\detect\fine_tune_ft6\results.csv",
        "label": "fine_tune(epochs=80)"
    },
        {
        "path": r"D:\dataset\AIhub\runs\detect\fine_tune_ft7\results.csv",
        "label": "hyp fine_tune(epochs=80)"
    },
]

# 2) 어떤 지표를 그릴지 선택
#   - (표시용 라벨, csv 안에서 찾을 수 있는 컬럼 후보들)
metrics = [
    ("Precision",      ["metrics/precision(B)", "metrics/precision"]), 
    ("Val Box Loss",   ["val/box_loss"]),
    ("Val DFL Loss",   ["val/dfl_loss"]),
    ("Recall",         ["metrics/recall(B)", "metrics/recall"]),
    ("mAP50",          ["metrics/mAP50(B)", "metrics/mAP50"]),
    ("mAP50-95",       ["metrics/mAP50-95(B)", "metrics/mAP50-95"]),
]

# 3) csv들을 읽어서 DataFrame으로 저장
for info in csv_infos:
    info["df"] = pd.read_csv(info["path"])

# 4) 그래프 그리기
num_metrics = len(metrics)
fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4), squeeze=False)
axes = axes[0]  # 1행이니까

for i, (metric_name, candidates) in enumerate(metrics):
    ax = axes[i]

    for info in csv_infos:
        df = info["df"]
        label = info["label"]

        # 이 metric에 대해 실제로 존재하는 컬럼 이름 찾기
        col_name = None
        for c in candidates:
            if c in df.columns:
                col_name = c
                break

        if col_name is None:
            print(f"[WARN] {metric_name} ({candidates}) 컬럼이 {info['path']}에 없음, 스킵")
            continue

        ax.plot(df["epoch"], df[col_name],
                marker="o", linewidth=1, markersize=3, label=label)

    ax.set_title(metric_name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_name)
    ax.grid(True)
    ax.legend()

plt.tight_layout()

# ⭐⭐ 그래프 저장 경로 설정 ⭐⭐
save_path = r"D:\dataset\AIhub\comparison_plot.png"

# ⭐⭐ 그래프 저장 ⭐⭐
plt.savefig(save_path, dpi=300)
print(f"그래프 저장 완료 → {save_path}")

plt.show()

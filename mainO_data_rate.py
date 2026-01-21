# setup

# 사용할 라이브러리 정리
import os                      # 파일/폴더 경로 처리, 디렉토리 목록 조회
import numpy as np             # 수치 계산
import pandas as pd            # CSV 로딩 및 데이터프레임 처리
import kagglehub               # Kaggle 데이터셋 다운로드
import matplotlib.pyplot as plt  # 시각화(기본)
import matplotlib.font_manager as fm  # 한글 폰트 설정용
import seaborn as sns          # 시각화(고급)

# 데이터셋 다운로드
path = kagglehub.dataset_download("ziya07/smart-manufacturing-iot-cloud-monitoring-dataset")
print("Path to dataset files:", path)

# csv 파일 불러오기
files = os.listdir(path)
csv_file = [f for f in files if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

# ---- 한글 폰트 설정 ----#
font_path = r"C:\Windows\Fonts\malgun.ttf"
fp = fm.FontProperties(fname=font_path)

# ====== columns ======
STATUS_COL = "machine_status"
MAINT_COL  = "maintenance_required"
ANOM_COL   = "anomaly_flag"
RISK_COL   = "downtime_risk"
PRED_COL   = "predicted_remaining_life"

d = df.copy()

# dtype 정리
for c in [MAINT_COL, ANOM_COL, RISK_COL]:
    d[c] = d[c].astype(int)

# ====== 1) machine_status × maintenance_required (rate %) ======
ct = pd.crosstab(d[STATUS_COL], d[MAINT_COL]).reindex(
    index=sorted(d[STATUS_COL].dropna().unique()),
    columns=[0, 1], fill_value=0
)
ct_ratio = ct.div(ct.sum(axis=1), axis=0) * 100

# ====== 2) P(maint=1 | anomaly_flag), 3) P(maint=1 | downtime_risk) ======
p_maint_given_anom = (d.groupby(ANOM_COL)[MAINT_COL].mean().reindex([0, 1]) * 100)
p_maint_given_risk = (d.groupby(RISK_COL)[MAINT_COL].mean().reindex([0, 1]) * 100)

# ====== (D) RUL 히스토그램 bins ======
bin_width = 10
max_v = np.ceil(d[PRED_COL].max() / bin_width) * bin_width
bins = np.arange(0, max_v + bin_width, bin_width)

# ======  그래프 시각화  ======
TITLE_FS = 12
LABEL_FS = 10
TICK_FS  = 9
LEG_FS   = 9

fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)

for ax in axes.flat:
    ax.tick_params(axis="both", labelsize=TICK_FS)

# (A) machine_status별 유지보수 비율
ax = axes[0, 0]
ct_ratio.plot(kind="bar", stacked=True, ax=ax, width=0.75, color=["C0", "C1"], legend=False)
ax.set_title("machine_status별 유지보수 비율", fontproperties=fp, fontsize=TITLE_FS)
ax.set_xlabel("machine_status", fontsize=LABEL_FS)
ax.set_ylabel(" maintenance rate [%]", fontsize=LABEL_FS)
ax.tick_params(axis="x", labelrotation=20)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=["maintenance = 0", "maintenance = 1"],
          fontsize=LEG_FS, loc="upper right", frameon=True)
ax.grid(False)

# (B) anomaly_flag별 유지보수 비율
ax = axes[0, 1]
p_maint_given_anom.plot(kind="bar", ax=ax, width=0.7, color=["C0", "C1"])
ax.set_title("anomaly_flag별 유지보수 비율", fontproperties=fp, fontsize=TITLE_FS)
ax.set_xlabel("anomaly_flag (0/1)", fontsize=LABEL_FS)
ax.set_ylabel("maintenance rate [%]", fontsize=LABEL_FS)
ax.set_ylim(0, 100)
ax.set_xticklabels(["0", "1"], rotation=0)
ax.legend(handles=handles, labels=["maintenance = 0", "maintenance = 1"],
          fontsize=LEG_FS, loc="upper right", frameon=True)
ax.grid(False)

# (C) downtime_risk별 유지보수 비율
ax = axes[1, 0]
p_maint_given_risk.plot(kind="bar", ax=ax, width=0.7, color=["C0", "C1"])
ax.set_title("downtime_risk별 유지보수 비율", fontproperties=fp, fontsize=TITLE_FS)
ax.set_xlabel("downtime_risk (0/1)", fontsize=LABEL_FS)
ax.set_ylabel("maintenance rate [%]", fontsize=LABEL_FS)
ax.set_ylim(0, 100)
ax.set_xticklabels(["0", "1"], rotation=0)
ax.legend(handles=handles, labels=["maintenance = 0", "maintenance = 1"],
          fontsize=LEG_FS, loc="upper right", frameon=True)
ax.grid(False)

# (D) RUL 히스토그램
ax = axes[1, 1]
ax.hist(d.loc[d[MAINT_COL] == 0, PRED_COL], bins=bins, alpha=0.5, edgecolor="black",
        label="maintenance = 0")
ax.hist(d.loc[d[MAINT_COL] == 1, PRED_COL], bins=bins, alpha=0.5, edgecolor="black",
        label="maintenance = 1")
ax.set_title("RUL 분포 비교", fontproperties=fp, fontsize=TITLE_FS)
ax.set_xlabel(PRED_COL, fontsize=LABEL_FS)
ax.set_ylabel("count", fontsize=LABEL_FS)
ax.legend(fontsize=LEG_FS, loc="upper right", frameon=True)
ax.grid(False)

fig.suptitle("Maintenance relationships (status / anomaly / downtime_risk / RUL)",
             y=1.02, fontproperties=fp, fontsize=13)

# 레이아웃 정리 후 출력
plt.tight_layout()  # 범례/라벨이 잘리지 않도록 여백 자동 조정
plt.show()          # 그래프 표시


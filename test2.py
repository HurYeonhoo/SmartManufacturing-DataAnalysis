import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 불러오기
df = pd.read_csv('smart_manufacturing_data.csv')


# 2. 스타일 및 전체 폰트 크기 설정 (기존 15 -> 10으로 축소)
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 10 

# 2행 3열 도화지 생성
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.suptitle('Smart Manufacturing IoT Data Analysis Report', fontsize=16, fontweight='bold')

# --- (1) Correlation Heatmap ---
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='RdYlGn', ax=axes[0,0], fmt=".2f", annot_kws={"size": 8})
axes[0,0].set_title('1. Sensor Correlation Map', fontsize=11)

# --- (2) Box Plot: Temperature ---
sns.boxplot(x='maintenance_required', y='temperature', data=df, ax=axes[0,1], palette='Set2')
axes[0,1].set_title('2. Temp Distribution by Maintenance', fontsize=11)

# --- (3) Pie Chart: Failure Type ---
failure_counts = df['failure_type'].value_counts()
axes[0,2].pie(failure_counts, labels=failure_counts.index, autopct='%1.1f%%', 
               startangle=140, colors=sns.color_palette('pastel'), textprops={'fontsize': 8})
axes[0,2].set_title('3. Ratio of Failure Types', fontsize=11)

# --- (4) Count Plot: Machine Status ---
sns.countplot(x='machine_status', hue='maintenance_required', data=df, ax=axes[1,0])
axes[1,0].set_title('4. Status Count by Maintenance', fontsize=11)

# --- (5) Scatter Plot: Temp vs Vibration ---
sns.scatterplot(x='temperature', y='vibration', hue='maintenance_required', 
                data=df, ax=axes[1,1], alpha=0.4, s=20)
axes[1,1].set_title('5. Temp vs Vibration Analysis', fontsize=11)

# --- (6) Bar Plot: Energy Consumption ---
sns.barplot(x='failure_type', y='energy_consumption', data=df, ax=axes[1,2], palette='viridis')
axes[1,2].set_title('6. Avg Energy by Failure Type', fontsize=11)

# 이미지 저장 및 출력
plt.savefig('test2_report.png', dpi=300, bbox_inches='tight')
plt.show()
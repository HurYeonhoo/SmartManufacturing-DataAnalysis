# =================================================================================
# 파일명: 02_Maintenance_Action_Plan_Generator.py
# 설명: 현장 정비팀을 위한 데이터 기반의 구체적인 정비 지침 이미지 및 점검 리스트 생성
# 생성 폴더: 02_Maintenance_Action_Plan_Generator_image
# 주요 결과물: 01~05 개별 분석 차트, 06 정비 지침서(PNG), 긴급 점검 리스트(CSV)
# =================================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# [1] 한글 설정 및 저장 폴더 생성
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
save_dir = '02_Maintenance_Action_Plan_Generator_image'
os.makedirs(save_dir, exist_ok=True)

# [2] 데이터 로드 및 전처리
df = pd.read_csv('smart_manufacturing_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 기계별 최신 상태 데이터 추출 및 잔여수명 기준 그룹화
latest_data = df.sort_values('timestamp').groupby('machine_id').last()

def classify_status(rul):
    if rul <= 15: return 'Critical(위험)'
    elif rul <= 45: return 'Warning(주의)'
    else: return 'Normal(정상)'
df['status_group'] = df['predicted_remaining_life'].apply(classify_status)

# --- [3] 개별 분석 차트 생성 및 저장 (생략 없이 모두 포함) ---

# 01. 센서상관관계_지도: 어떤 데이터들이 서로 연관되어 있는지 분석
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".2f")
plt.title('01. 센서 상관관계 분석 지도', fontweight='bold')
plt.savefig(f'{save_dir}/01_센서상관관계_지도.png', dpi=300)
plt.close()

# 02. 유지보수대상_온도분포: 정비 필요군과 정상군의 온도 차이 비교
plt.figure(figsize=(8, 6))
sns.boxplot(x='maintenance_required', y='temperature', data=df, palette='Set2')
plt.title('02. 유지보수 필요 여부별 온도 분포', fontweight='bold')
plt.savefig(f'{save_dir}/02_유지보수대상_온도분포.png', dpi=300)
plt.close()

# 03. 고장원인_비중분석: 정상 상태를 제외한 실제 고장 원인 비중
plt.figure(figsize=(8, 6))
fail_counts = df[df['failure_type'] != 'Normal']['failure_type'].value_counts()
plt.pie(fail_counts, labels=fail_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('03. 설비 고장 원인별 비중 분석', fontweight='bold')
plt.savefig(f'{save_dir}/03_고장원인_비중분석.png', dpi=300)
plt.close()

# 04. 기계상태_운영현황: 기계 운영 상태(On/Off)와 정비 필요성의 관계
plt.figure(figsize=(8, 6))
sns.countplot(x='machine_status', hue='maintenance_required', data=df, palette='viridis')
plt.title('04. 기계 운영 상태별 정비 현황', fontweight='bold')
plt.savefig(f'{save_dir}/04_기계상태_운영현황.png', dpi=300)
plt.close()

# 05. 위험그룹별_진동차이: 잔여 수명 그룹에 따른 진동 수치 분석
plt.figure(figsize=(10, 7))
sns.boxplot(x='status_group', y='vibration', data=df, order=['Normal(정상)', 'Warning(주의)', 'Critical(위험)'], 
            palette={'Normal(정상)':'#2ca02c', 'Warning(주의)':'#ff7f0e', 'Critical(위험)':'#d62728'})
plt.title('05. 위험 단계별 진동 수치 분석', fontweight='bold', color='darkred')
plt.savefig(f'{save_dir}/05_위험그룹별_진동차이.png', dpi=300)
plt.close()

# 06. 최종_정비전략_지침: 데이터 분석 결과 요약 및 현장 액션 플랜
plt.figure(figsize=(10, 6))
plt.axis('off')
critical_id = latest_data.nsmallest(1, 'predicted_remaining_life').index[0]
est_downtime = latest_data.nsmallest(5, 'predicted_remaining_life')['downtime_risk'].sum() * 24
summary_text = (
    " [ 스마트 팩토리 예방 정비 실행 지침 ]\n\n"
    f"▶ 최우선 점검 장비: 기계 번호 #{critical_id}\n"
    f"▶ 예상 생산 손실량: 약 {est_downtime:.1f} 시간 발생 위험\n\n"
    " [ 주요 지시 사항 ]\n"
    " 1. Critical 그룹 기계의 진동 수치가 비정상적으로 높음(Ref #05)\n"
    " 2. 해당 장비의 베어링 마모 및 모터 축 정렬 상태 즉시 점검\n"
    " 3. Warning 그룹 기계들의 정비 주기를 다음 주로 앞당길 것\n"
    " 4. 상세 기계별 잔여수명은 '긴급점검리스트.csv' 파일 참조"
)
plt.text(0.1, 0.5, summary_text, fontsize=14, family='monospace', bbox={'facecolor':'white', 'edgecolor':'black', 'pad':20})
plt.savefig(f'{save_dir}/06_최종_정비전략_지침.png', dpi=300, bbox_inches='tight')
plt.close()

# [4] 정비팀용 상세 리스트 CSV 파일로 별도 저장
latest_data.nsmallest(10, 'predicted_remaining_life').to_csv(f'{save_dir}/긴급점검리스트.csv', encoding='utf-8-sig')

print(f"✅ [작업 완료] 실무 지침서와 6개의 분석 차트, 점검 리스트(CSV)가 '{save_dir}' 폴더에 모두 저장되었습니다.")
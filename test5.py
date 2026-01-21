import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 환경 설정
plt.rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="whitegrid", palette="husl")

def run_comprehensive_analysis(file_path):
    if not os.path.exists(file_path):
        print("ERROR: File not found")
        return

    # 데이터 로드
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    binary_cols = ['anomaly_flag', 'downtime_risk', 'maintenance_required']
    df[binary_cols] = df[binary_cols].astype('int8')

    # 폴더 생성
    save_dir = '📊_Report'
    os.makedirs(save_dir, exist_ok=True)

    charts = []

    # ===== 01. Failure Type =====
    plt.figure(figsize=(12, 10))
    fail_df = df[df['failure_type'] != 'Normal']
    fail_counts = fail_df['failure_type'].value_counts()
    plt.pie(fail_counts, labels=fail_counts.index, autopct='%1.1f%%', startangle=140, 
            explode=[0.08]*len(fail_counts), colors=sns.color_palette('pastel'))
    plt.title('01. Failure Type Distribution')
    plt.savefig(f'{save_dir}/01_Failure_Type.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    charts.append("01_Failure_Type.png")

    # ===== 02. Correlation Matrix =====
    plt.figure(figsize=(14, 12))
    corr = df.select_dtypes([np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdYlGn', fmt='.2f')
    plt.title('02. Sensor Correlation Matrix')
    plt.savefig(f'{save_dir}/02_Correlation_Matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    charts.append("02_Correlation_Matrix.png")

    # ===== 03. Sensor Distributions =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('03. Sensor Distributions')
    
    sns.histplot(df['temperature'], kde=True, ax=axes[0,0], color='coral')
    axes[0,0].set_title('Temperature')
    
    sns.histplot(df['vibration'], kde=True, ax=axes[0,1], color='skyblue')
    axes[0,1].set_title('Vibration')
    
    sns.histplot(df['energy_consumption'], kde=True, ax=axes[1,0], color='gold')
    axes[1,0].set_title('Energy')
    
    sns.histplot(df['pressure'], kde=True, ax=axes[1,1], color='lightgreen')
    axes[1,1].set_title('Pressure')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/03_Sensor_Distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    charts.append("03_Sensor_Distributions.png")

    # ===== 04. Machine Risk =====
    plt.figure(figsize=(16, 8))
    risk_by_machine = df.groupby('machine_id')['downtime_risk'].mean().sort_values(ascending=False)
    colors = ['red' if i < 10 else 'lightcoral' for i in range(len(risk_by_machine))]
    risk_by_machine.plot(kind='bar', color=colors, alpha=0.8)
    plt.title('04. Machine Risk Ranking')
    plt.xlabel('Machine ID')
    plt.ylabel('Risk Score')
    plt.xticks(rotation=0)
    plt.savefig(f'{save_dir}/04_Machine_Risk.png', dpi=300, bbox_inches='tight')
    plt.close()
    charts.append("04_Machine_Risk.png")

    # ===== 05. Lead Time =====
    def calc_lead_time(df):
        lead_times = []
        for mid, group in df.groupby('machine_id'):
            risk_times = group[group['downtime_risk'] == 1]['timestamp']
            maint_times = group[group['maintenance_required'] == 1]['timestamp']
            for mt in maint_times:
                prev_risk = risk_times[risk_times < mt]
                if len(prev_risk) > 0:
                    lead_time = (mt - prev_risk.iloc[-1]).total_seconds() / 60
                    lead_times.append(lead_time)
        return np.array(lead_times)

    lead_times = calc_lead_time(df)
    plt.figure(figsize=(12, 8))
    plt.hist(lead_times, bins=30, alpha=0.7, color='purple')
    plt.title('05. Lead Time Distribution')
    plt.xlabel('Minutes')
    plt.ylabel('Frequency')
    plt.savefig(f'{save_dir}/05_Lead_Time.png', dpi=300, bbox_inches='tight')
    plt.close()
    charts.append("05_Lead_Time.png")

    # ===== 06. Status vs Temperature (Violin) =====
    plt.figure(figsize=(14, 10))
    sns.violinplot(data=df, x='machine_status', y='temperature', 
                   hue='maintenance_required', split=True, inner='quartile', palette='Set2')
    plt.title('06. Temperature by Machine Status')
    plt.xlabel('Machine Status')
    plt.ylabel('Temperature')
    plt.savefig(f'{save_dir}/06_Status_Temperature.png', dpi=300, bbox_inches='tight')
    plt.close()
    charts.append("06_Status_Temperature.png")

    # ===== 07. TOP5 Risky Machines =====
    top5_risk = risk_by_machine.head(5).index
    df_top5 = df[df['machine_id'].isin(top5_risk)]
    plt.figure(figsize=(16, 10))
    for mid in top5_risk:
        machine_data = df_top5[df_top5['machine_id'] == mid].groupby('timestamp')['temperature'].mean()
        plt.plot(machine_data.index, machine_data.values, label=f'Machine #{mid}', linewidth=2.5)
    plt.title('07. TOP5 Riskiest Machines Trend')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/07_TOP5_Riskiest_Machines.png', dpi=300, bbox_inches='tight')
    plt.close()
    charts.append("07_TOP5_Riskiest_Machines.png")

    # ===== 08. Dashboard =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('08. Executive Dashboard')
    
    sns.boxplot(data=df, x='maintenance_required', y='vibration', ax=axes[0,0])
    axes[0,0].set_title('Vibration by Maintenance')
    
    risk_by_machine.head(10).plot(kind='bar', ax=axes[0,1], color='red')
    axes[0,1].set_title('Top 10 Risky Machines')
    
    fail_counts.plot(kind='pie', autopct='%1.1f%%', ax=axes[1,0])
    axes[1,0].set_title('Failure Types')
    
    corr_small = df[['temperature','vibration','downtime_risk']].corr()
    sns.heatmap(corr_small, annot=True, ax=axes[1,1], cmap='coolwarm')
    axes[1,1].set_title('Key Correlations')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/08_Executive_Dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    charts.append("08_Executive_Dashboard.png")

    # 최종 출력 (깔끔하게)
    print("SUCCESS")
    print(f"FOLDER: {save_dir}")
    print("CHARTS:")
    for chart in charts:
        print(chart)
    print("COMPLETE")

if __name__ == "__main__":
    run_comprehensive_analysis('smart_manufacturing_data.csv')

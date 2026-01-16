import os
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
# 데이터셋 다운로드
path = kagglehub.dataset_download(
    "ziya07/smart-manufacturing-iot-cloud-monitoring-dataset"
)

print("Path to dataset files:", path)

# csv 파일 불러오기 
files = os.listdir(path)
csv_file = [f for f in files if f.endswith('.csv')][0]

df = pd.read_csv(os.path.join(path, csv_file))


#maintenance_required가 1인 값을 추출
df_maint_1 = df[df['maintenance_required'] == 1]
#디버깅용
# print(df_maint_1.shape)
# df_maint_1.head()

# 점검이 필요한 상태일 때, 온도의 값별 빈도 수 표현
temp = df[df['maintenance_required'] == 1]['temperature']
# 평균온도 구하기
mean_temp = temp.mean()
# 평균온도 출력(디버깅용)
# print(f"Average temperature: {temp_maint_1.mean():.2f}") #소수점 둘째자리까지만 출력


# 점검이 필요한 상태일 때, 진동의 값별 빈도 수 표현
vib = df[df['maintenance_required'] == 1]['vibration']
# 점검 필요 시, 진동 평균 값 
mean_vibration = vib.mean()
# 평균온도 출력(디버깅용)
#print(f"Average vibration (maintenance_required = 1): {mean_vibration:.2f}")

# 점검이 필요한 상태일 때, 습도값별 빈도 수 표현
humid = df[df['maintenance_required'] == 1]['humidity']

# 점검이 필요한 상태일 때, 압력값별 빈도 수 표현
press = df[df['maintenance_required'] == 1]['pressure']

# 점검이 필요한 상태일 때, 에너지소비 값별 빈도 수 표현
energy= df[df['maintenance_required'] == 1]['energy_consumption']






# 그래프 그리기
plt.figure()
# 바이올릿 플롯으로 그래프 표현
plt.violinplot(
    [temp.values, vib.values, humid.values],
    showmeans=True
)

plt.xticks([1, 2], ['Temperature', 'Vibration', 'Humidity'])
plt.title("Violin plot of Temperature and Vibration\n(maintenance_required = 1)")
plt.ylabel("Value")
plt.show()


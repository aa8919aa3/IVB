import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 定義參數
Ic = 1.0e-6          # 臨界電流
phi_0 = np.pi / 4 # 相位偏移
f = 3e5           # 頻率
T = 0.8           # 參數T
k = -0.01           # 二次項係數
r = 50e-3          # 線性項係數
C = 10.0e-6           # 常數項
d = -10.0e-3           # 偏移量

# 定義範圍
Phi_ext = np.linspace(-30e-6, -10e-6, 1001)  # 外部磁通範圍

# 定義函數
def I_s(Phi_ext):
    term1 = Ic * np.sin(2 * np.pi * f * (Phi_ext - d) - phi_0)
    term2 = np.sqrt(1 - T * np.sin((2 * np.pi * f * (Phi_ext - d) - phi_0) / 2)**2)
    term3 = k * (Phi_ext - d)**2
    term4 = r * (Phi_ext - d)
    return term1 / term2 + term3 + term4 + C

# 計算函數值
I_values = I_s(Phi_ext)

# 添加雜訊
noise_level = 1e-7 # 雜訊強度
noise = noise_level * np.random.normal(size=Phi_ext.shape)
I_values_noisy = I_values + noise

# 輸出模擬參數成csv文件
params_df = pd.DataFrame({
    'Ic': [Ic],
    'phi_0': [phi_0],
    'f': [f],
    'T': [T],
    'k': [k],
    'r': [r],
    'C': [C],
    'd': [d]
})
params_df.to_csv('simulation_parameters.csv', index=False)

# 輸出模擬結果成csv文件
results_df = pd.DataFrame({
    'Phi_ext': Phi_ext,
    'I_s': I_values_noisy
})
results_df.to_csv('simulation_results.csv', index=False)

# 繪製結果
plt.figure(figsize=(10, 6))
plt.subplots_adjust(bottom=0.15)  # Make room for parameters
plt.plot(Phi_ext, I_values, label="Original Function", linewidth=2)
plt.plot(Phi_ext, I_values_noisy, label="Noisy Data", linestyle='--', alpha=0.7)
plt.xlabel(r"$\Phi_{ext}$")
plt.ylabel(r"$I_s$")
plt.legend()
params_part = rf"Ic={Ic:.2e}, $\phi_0$={phi_0:.2e}, f={f:.2e}, T={T:.2%}, k={k:.2e}, r={r:.2e}, C={C:.2e}, d={d:.2e}"
plt.title(r"Simulation of $I_s = \frac{I_c \sin \left(2 \pi f\left(\Phi_{ext}-d\right)-\phi_0\right)}{\sqrt{1-T \sin ^2\left(\frac{2 \pi f\left(\Phi_{ext}-d\right)-\phi_0}{2}\right)}}+k\left(\Phi_{ext}-d\right)^2+r\left(\Phi_{ext}-d\right)+C$ with Noise")
plt.figtext(0.5, 0.02, params_part, ha='center', va='bottom', fontsize=8, )
plt.grid()
plt.show()

# 定義簡單正弦函數
def I_s_sine(Phi_ext):
    term1 = Ic * np.sin(2 * np.pi * f * (Phi_ext - d) - phi_0)
    term3 = k * (Phi_ext - d)**2
    term4 = r * (Phi_ext - d)
    return term1 + term3 + term4 + C

# 計算函數值
I_values = I_s_sine(Phi_ext)
noise = noise_level * np.random.normal(size=Phi_ext.shape)
I_values_noisy = I_values + noise

# 輸出模擬參數成csv文件
params_df = pd.DataFrame({
    'Ic': [Ic],
    'phi_0': [phi_0],
    'f': [f],
    'T': None,  # T不適用於簡單正弦函數
    'k': [k],
    'r': [r],
    'C': [C],
    'd': [d]
})
params_df.to_csv('sine_simulation_parameters.csv', index=False)
# 輸出模擬結果成csv文件
results_df = pd.DataFrame({
    'Phi_ext': Phi_ext,
    'I_s': I_values_noisy
})
results_df.to_csv('sine_simulation_results.csv', index=False)

# 繪製結果
plt.figure(figsize=(10, 6))
plt.subplots_adjust(bottom=0.15)  # Make room for parameters
plt.plot(Phi_ext, I_values, label="Original Function (Sine)", linewidth=2)
plt.plot(Phi_ext, I_values_noisy, label="Noisy Data", linestyle='--', alpha=0.7)
plt.xlabel(r"$\Phi_{ext}$")
plt.ylabel(r"$I_s$")
plt.legend()
params_part = rf"Ic={Ic:.2e}, $\phi_0$={phi_0:.2e}, f={f:.2e}, k={k:.2e}, r={r:.2e}, C={C:.2e}, d={d:.2e}"
plt.title(r"Simulation of $I_s = I_c \sin(2 \pi f(\Phi_{ext}-d)-\phi_0)+k(\Phi_{ext}-d)^2+r(\Phi_{ext}-d)+C$ with Noise")
plt.figtext(0.5, 0.02, params_part, ha='center', va='bottom', fontsize=8,)
plt.grid()
plt.show()






















































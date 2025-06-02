import pandas as pd
import matplotlib.pyplot as plt
dataid = 93
# 讀取數據
df = pd.read_csv(f'Ic/{dataid}Ic+.csv')
print(f"📊 Columns: {list(df.columns)}")
print(f"📈 Shape: {df.shape}")
# 簡潔的線圖
df.plot(x='y_field', y='Ic', figsize=(16, 9), 
        title='y_field vs Ic', marker='o', linewidth=2)
plt.grid(True, alpha=0.3)
plt.show()



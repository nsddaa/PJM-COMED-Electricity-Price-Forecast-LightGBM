import pandas as pd

import matplotlib

matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端弹窗显示
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb


# 1.1 读取电价+预测负荷数据（PJM.csv）
df_price = pd.read_csv('./datasets/PJM.csv')
# 重命名列，统一格式
df_price.rename(columns={
    'Date': 'timestamp',
    'Zonal COMED price': 'price',
    'System load forecast': 'pjm_system_load_forecast',
    'Zonal COMED load foecast': 'comed_load_forecast'
}, inplace=True)

# 1.2 读取实际负荷数据（COMED_hourly.csv）
df_actual = pd.read_csv('./datasets/market_data.csv')
df_actual.rename(columns={
    'timestamp': 'timestamp',  # 原列名就是timestamp，保留
    'COMED_MW': 'comed_load_actual'
}, inplace=True)

# 1.3 读取天气数据（weather.csv）
df_weather = pd.read_csv('./datasets/temperature.csv')
# 天气数据重命名
df_weather.rename(columns={
    'timestamp': 'timestamp',
    'Chicago': 'chicago_temp_f'
}, inplace=True)

#显式转换时间
df_price['timestamp'] = pd.to_datetime(df_price['timestamp'], errors='coerce')
df_actual['timestamp'] = pd.to_datetime(df_actual['timestamp'], errors='coerce')
df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'], errors='coerce')

# ==========================================
# 3. 只保留核心列，避免合并冲突
# ==========================================
df_price = df_price[['timestamp', 'price', 'pjm_system_load_forecast', 'comed_load_forecast']]
df_actual = df_actual[['timestamp', 'comed_load_actual']]
df_weather = df_weather[['timestamp', 'chicago_temp_f']]

# ======================
# 3. 【关键】共同区间：2013-01-01 ~ 2017-12-31
# ======================
start = '2013-01-01'
end = '2017-12-31 23:00:00'

df_price = df_price.loc[df_price['timestamp'].between(start, end)]
df_actual = df_actual.loc[df_actual['timestamp'].between(start, end)]
df_weather = df_weather.loc[df_weather['timestamp'].between(start, end)]

# --------------------------
# 5. 排序 + 去重（必须）
# --------------------------
def clean(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset='timestamp', keep='first')
    return df

df_price = clean(df_price)
df_actual = clean(df_actual)
df_weather = clean(df_weather)

# --------------------------
# 6. 三表合并
# --------------------------
df = df_price.merge(df_actual, on='timestamp', how='inner')
df = df.merge(df_weather, on='timestamp', how='inner')

# --------------------------
# 7. 生成完整连续时间序列（最关键）
# --------------------------
full_idx = pd.date_range(start=start, end=end, freq='h')
df = df.set_index('timestamp').reindex(full_idx)
df = df.reset_index().rename(columns={'index':'timestamp'})

# --------------------------
# 8. 特征工程
# --------------------------
df['temp_C'] = (df['chicago_temp_f'] - 32) *5/9
df['load_error'] = df['comed_load_actual'] - df['comed_load_forecast']

df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month
df['weekday'] = df['timestamp'].dt.weekday
df['is_weekend'] = (df['weekday'] >=5).astype(int)

# --------------------------
# 9. 最终排序
# --------------------------
df = df.sort_values('timestamp').reset_index(drop=True)

# --------------------------
# 导出
# --------------------------
df.to_csv('./datasets/final_dataset.csv', index=False, encoding='utf-8-sig')


# ==========================
# 1. 读取你刚刚合并好的完美数据
# ==========================
df = pd.read_csv('./datasets/final_dataset.csv', parse_dates=['timestamp'])
# 构造历史滞后特征（无未来泄露）
for i in [24, 48, 72]:
    df[f'price_lag_{i}'] = df['price'].shift(i)
    df[f'load_lag_{i}'] = df['comed_load_forecast'].shift(i)

df['price_rolling_mean_24'] = df['price'].rolling(24).mean()
df['price_rolling_max_24'] = df['price'].rolling(24).max()
df['load_rolling_mean_24'] = df['comed_load_forecast'].rolling(24).mean()


df = df.dropna()

features = [
    'comed_load_forecast',
    'pjm_system_load_forecast',
    'chicago_temp_f',
    'temp_C',
    'hour',
    'month',
    'weekday',
    'is_weekend',
    'price_lag_24',
    'price_lag_48',
    'price_lag_72',
    'load_lag_24',
    'load_lag_48',
    'load_lag_72',
    'price_rolling_max_24',
    'load_rolling_mean_24'
    ]
# 目标：预测电价
X = df[features]
y = df['price']

# ==========================
# 3. 时间序列不能随机划分！按时间切分
# ==========================
train_size = int(len(df) * 0.95)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# ==========================
# 4. LightGBM 模型
# ==========================
model = lgb.LGBMRegressor(
    n_estimators=1000,  #决策树的数量
    learning_rate=0.03,  #控制每棵树对模型的贡献权重
    max_depth=6,   #单棵树的最大深度，控制树的复杂度，防止过拟合
    num_leaves=31, #单棵树的最大叶子节点数
    subsample=0.8, #
    reg_alpha=0.1,  # L1正则
    reg_lambda=0.1, # L2正则
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# 5. 预测 & 评估
# ==========================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("✅ 模型训练完成！")
print(f"RMSE = {rmse:.2f}")
print(f"MAE  = {mae:.2f}")

# ==========================
# 6. 特征重要性
# ==========================
plt.figure(figsize=(10,5))
lgb.plot_importance(model, max_num_features=10)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ==========================
# 7. 预测图
# ==========================

plt.figure(figsize=(14,5))
plt.plot(y_test.values[:200], label='Actual Price', alpha=0.8)
plt.plot(y_pred[:200], label='Predicted Price', alpha=0.8)
plt.legend()
plt.title("Price Prediction vs Actual")
plt.show()
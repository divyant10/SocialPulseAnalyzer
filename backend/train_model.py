import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'mock_social_virality_dataset_extended.csv')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'virality_model.pkl')

df = pd.read_csv(DATA_PATH)

le = LabelEncoder()
df['platform_encoded'] = le.fit_transform(df['platform'])

df['log_likes'] = np.log1p(df['likes'])
df['log_views'] = np.log1p(df['views'])
df['engagement_rate'] = (df['likes'] + df['views']) / (df['caption_length'] + 1)
df['hashtag_count'] = df['hashtags'].apply(lambda x: len(str(x).split()))

X = df[['platform_encoded', 'log_likes', 'log_views', 'caption_length', 'sentiment', 'engagement_rate', 'hashtag_count']]
y = df['virality_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained.")
print(f"📉 MSE: {mse:.2f}")
print(f"📏 MAE: {mae:.2f}")
print(f"📈 R² Score: {r2:.2f}")

joblib.dump(model, MODEL_PATH)
print(f"📦 Model saved to {MODEL_PATH}")

importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

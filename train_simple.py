import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.pipeline import Pipeline

print("=" * 70)
print("TRAINING SLEEP QUALITY MODEL (WITHOUT SLEEP STAGES)")
print("=" * 70)

print("\nLoading data...")
df = pd.read_csv('sleep_clean.csv')
print(f"✓ Loaded {len(df)} rows")

# ONLY USE FEATURES USERS CAN ACTUALLY PROVIDE
feature_names = [
    'Age',
    'Sleep duration',
    'Awakenings',
    'Caffeine consumption',
    'Alcohol consumption',
    'Exercise frequency',
    'Gender_Male',
    'Smoking status_Yes'
]

X = df[feature_names]
y = df['Sleep efficiency']

print(f"\n✓ Using {len(feature_names)} user-provided features")
print("  Features:", feature_names)

print("\nTraining ensemble model...")
print("  This may take 30-60 seconds...")

# Create ensemble model
elastic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elastic', ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42))
])

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model = StackingRegressor(
    estimators=[
        ('elastic_net', elastic_pipeline),
        ('random_forest', rf_model)
    ],
    final_estimator=Ridge(alpha=0.1),
    cv=5
)

# Train model
model.fit(X, y)
print("✓ Model trained successfully!")

# Test prediction
test_sample = X.iloc[0:1]
test_pred = model.predict(test_sample)[0]
actual = y.iloc[0]
print(f"✓ Test prediction: {test_pred:.3f} ({test_pred*100:.1f}%) - Actual: {actual:.3f}")

# Save model
print("\nSaving model files...")
with open('sleep_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)

scaler = StandardScaler()
scaler.fit(X)
with open('scaler_v2.pkl', 'wb') as f:
    pickle.dump(scaler, f)

import os
print(f"✓ Model saved: sleep_model_v2.pkl ({os.path.getsize('sleep_model_v2.pkl'):,} bytes)")
print(f"✓ Scaler saved: scaler_v2.pkl ({os.path.getsize('scaler_v2.pkl'):,} bytes)")

print("\n" + "=" * 70)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 70)
print("\nFeatures used (8 total):")
for i, feat in enumerate(feature_names, 1):
    print(f"  {i}. {feat}")
print("\n✅ Ready to deploy to Railway!")
print("=" * 70)

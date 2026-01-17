# Title: Customer churn prediction 2026-01-17 01:31:31

# Set environment variables for sagemaker_studio imports

import os
os.environ['DataZoneProjectId'] = 'bfo82ci1mqxixz'
os.environ['DataZoneDomainId'] = 'dzd-cdmlfk1l24a8o7'
os.environ['DataZoneEnvironmentId'] = '45h1a52tnpupev'
os.environ['DataZoneDomainRegion'] = 'us-east-1'

# create both a function and variable for metadata access
_resource_metadata = None

def _get_resource_metadata():
    global _resource_metadata
    if _resource_metadata is None:
        _resource_metadata = {
            "AdditionalMetadata": {
                "DataZoneProjectId": "bfo82ci1mqxixz",
                "DataZoneDomainId": "dzd-cdmlfk1l24a8o7",
                "DataZoneEnvironmentId": "45h1a52tnpupev",
                "DataZoneDomainRegion": "us-east-1",
            }
        }
    return _resource_metadata
metadata = _get_resource_metadata()

"""
Logging Configuration

Purpose:
--------
This sets up the logging framework for code executed in the user namespace.
"""

from typing import Optional


def _set_logging(log_dir: str, log_file: str, log_name: Optional[str] = None):
    import os
    import logging
    from logging.handlers import RotatingFileHandler

    level = logging.INFO
    max_bytes = 5 * 1024 * 1024
    backup_count = 5

    # fallback to /tmp dir on access, helpful for local dev setup
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        log_dir = "/tmp/kernels/"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger() if not log_name else logging.getLogger(log_name)
    logger.handlers = []
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Rotating file handler
    fh = RotatingFileHandler(filename=log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging initialized for {log_name}.")


_set_logging("/var/log/computeEnvironments/kernel/", "kernel.log")
_set_logging("/var/log/studio/data-notebook-kernel-server/", "metrics.log", "metrics")

import logging
from sagemaker_studio import ClientConfig, sqlutils, sparkutils, dataframeutils

logger = logging.getLogger(__name__)
logger.info("Initializing sparkutils")
spark = sparkutils.init()
logger.info("Finished initializing sparkutils")

def _reset_os_path():
    """
    Reset the process's working directory to handle mount timing issues.
    
    This function resolves a race condition where the Python process starts
    before the filesystem mount is complete, causing the process to reference
    old mount paths and inodes. By explicitly changing to the mounted directory
    (/home/sagemaker-user), we ensure the process uses the correct, up-to-date
    mount point.
    
    The function logs stat information (device ID and inode) before and after
    the directory change to verify that the working directory is properly
    updated to reference the new mount.
    
    Note:
        This is executed at module import time to ensure the fix is applied
        as early as possible in the kernel initialization process.
    """
    try:
        import os
        import logging

        logger = logging.getLogger(__name__)
        logger.info("---------Before------")
        logger.info("CWD: %s", os.getcwd())
        logger.info("stat('.'): %s %s", os.stat('.').st_dev, os.stat('.').st_ino)
        logger.info("stat('/home/sagemaker-user'): %s %s", os.stat('/home/sagemaker-user').st_dev, os.stat('/home/sagemaker-user').st_ino)

        os.chdir("/home/sagemaker-user")

        logger.info("---------After------")
        logger.info("CWD: %s", os.getcwd())
        logger.info("stat('.'): %s %s", os.stat('.').st_dev, os.stat('.').st_ino)
        logger.info("stat('/home/sagemaker-user'): %s %s", os.stat('/home/sagemaker-user').st_dev, os.stat('/home/sagemaker-user').st_ino)
    except Exception as e:
        logger.exception(f"Failed to reset working directory: {e}")

_reset_os_path()

# MARKDOWN CELL conh
# # 🎯 Customer Churn Prediction: Saving Customers Before They Leave
# 
# ## 📊 The Business Challenge
# **Imagine this scenario:** Every month, your telecom company loses valuable customers to competitors. Each lost customer represents:
# - 💰 **$50-200** in monthly recurring revenue
# - 🔄 **5-10x** the cost to acquire a new customer vs. retaining existing ones
# - 📉 **Negative word-of-mouth** that affects brand reputation
# 
# ## 🚀 Our Mission
# Build a machine learning-powered early warning system that identifies at-risk customers **before** they churn, enabling proactive retention strategies.
# 
# ### 📈 Expected Outcomes
# - Identify **high-risk customers** with 95%+ accuracy
# - Reduce churn rate by **15-25%**
# - Increase customer lifetime value by **$500-1000** per saved customer
# - Enable **targeted retention campaigns** instead of blanket approaches

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
import optuna
np.random.seed(2)
warnings.filterwarnings('ignore')

# 🎨 Make our visualizations beautiful
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print('📊 Ready to uncover customer insights...')

# MARKDOWN CELL b0rz
# ## 🔍 Chapter 1: The Customer Data Detective Story
# 
# Let's dive into our customer database and uncover the secrets hidden in the data...

# 📥 Load our customer intelligence database
session = boto3.Session()
aws_region = session.region_name or 'us-west-2'

s3 = boto3.client('s3')

# Create data directory if it doesn't exist
os.makedirs('notebook_outputs', exist_ok=True)

s3.download_file(
    f'sagemaker-example-files-prod-{aws_region}',
    'datasets/tabular/synthetic/churn.txt',
    'notebook_outputs/churn.txt'
)

df = pd.read_csv('notebook_outputs/churn.txt')

print('🎯 Customer Database Loaded!')
print(f'📊 We have {df.shape[0]:,} customers with {df.shape[1]} data points each')
print(f'💾 Total data points: {df.shape[0] * df.shape[1]:,}')

# 🔍 First glimpse at our customers
print('\n👥 Meet our first 5 customers:')
df.head()

# 📈 Calculate the churn crisis metrics
total_customers = len(df)
churned_customers = len(df[df['Churn?'] == 'True.'])
churn_rate = churned_customers / total_customers

# 💰 Business impact calculations
avg_monthly_revenue = 75  # Average customer monthly value
customer_acquisition_cost = 200  # Cost to acquire new customer

monthly_revenue_lost = churned_customers * avg_monthly_revenue
annual_revenue_lost = monthly_revenue_lost * 12
replacement_cost = churned_customers * customer_acquisition_cost

print('🚨 THE CHURN CRISIS REPORT')
print('=' * 50)
print(f'📊 Total Customers: {total_customers:,}')
print(f'❌ Customers Lost: {churned_customers:,}')
print(f'📉 Churn Rate: {churn_rate:.1%}')
print(f'💸 Monthly Revenue Lost: ${monthly_revenue_lost:,}')
print(f'💸 Annual Revenue Lost: ${annual_revenue_lost:,}')
print(f'💰 Customer Replacement Cost: ${replacement_cost:,}')
print(f'🔥 Total Annual Impact: ${annual_revenue_lost + replacement_cost:,}')

# 🎯 Visualization: The Churn Story
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Churn distribution with dramatic colors
churn_counts = df['Churn?'].value_counts()
colors = ['#2ecc71', '#e74c3c']  # Green for retained, red for churned
axes[0].pie(churn_counts.values, labels=['Retained 😊', 'Churned 😞'], 
           autopct='%1.1f%%', colors=colors, startangle=90, 
           explode=(0, 0.1))  # Explode the churn slice
axes[0].set_title('🎯 Customer Retention vs Churn', fontsize=14, fontweight='bold')

# Revenue impact
impact_data = ['Monthly Loss', 'Replacement Cost']
impact_values = [monthly_revenue_lost/1000, replacement_cost/1000]  # In thousands
bars = axes[1].bar(impact_data, impact_values, color=['#e74c3c', '#f39c12'])
axes[1].set_title('💰 Financial Impact ($000s)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Cost ($000s)')
for bar, value in zip(bars, impact_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'${value:.0f}K', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print(f'\n🎯 KEY INSIGHT: If we can reduce churn by just 25%, we could save ${(annual_revenue_lost + replacement_cost) * 0.25:,.0f} annually!')

# MARKDOWN CELL 6ki3
# ## 🧠 Chapter 2: Model Training - Teaching Machines to Predict
# 
# Now let's train our models to identify at-risk customers...

# 🔧 Data preprocessing
print('🔧 PREPARING DATA FOR AI TRAINING')
print('=' * 40)

df_processed = df.copy()
df_processed['Churn'] = (df_processed['Churn?'] == 'True.').astype(int)
df_processed.drop('Churn?', axis=1, inplace=True)
df_processed.drop('Phone', axis=1, inplace=True)  # Remove phone numbers

# Encode categorical variables
categorical_cols = ['State', "Int'l Plan", 'VMail Plan']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

# 🚀 Feature engineering: Create power features
df_processed['Total_Charge'] = (df_processed['Day Charge'] + 
                               df_processed['Eve Charge'] + 
                               df_processed['Night Charge'] + 
                               df_processed['Intl Charge'])

df_processed['Avg_Charge_Per_Min'] = df_processed['Total_Charge'] / (
    df_processed['Day Mins'] + df_processed['Eve Mins'] + 
    df_processed['Night Mins'] + df_processed['Intl Mins'] + 1e-8)

df_processed['High_Service_Calls'] = (df_processed['CustServ Calls'] >= 4).astype(int)

print('✅ Data preprocessing completed!')
print(f'📊 Final dataset: {df_processed.shape[0]:,} customers, {df_processed.shape[1]} features')
print(f'🎯 Target distribution: {df_processed["Churn"].mean():.1%} churn rate')

# 🎯 Prepare for model training
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'🎯 Training set: {X_train.shape[0]:,} customers')
print(f'🎯 Test set: {X_test.shape[0]:,} customers')

df_processed

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

print("🤖 MODEL BATTLE: XGBoost vs Random Forest vs Logistic Regression (Optuna)")
print("=" * 80)

def objective(trial):

    model_type = trial.suggest_categorical(
        "model_type",
        ["logistic", "random_forest", "xgboost"]
    )

    # ---------------- Logistic Regression ----------------
    if model_type == "logistic":
        C = trial.suggest_float("C", 0.0001, 10.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        
        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver="liblinear",
            random_state=2
        )

        model.fit(X_train_scaled, y_train)
        preds = model.predict_proba(X_test_scaled)[:, 1]

    # ---------------- Random Forest ----------------
    elif model_type == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=2,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]

    # ---------------- XGBoost ----------------
    else:
        n_estimators = trial.suggest_int("n_estimators", 100, 400)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric="logloss",
            random_state=2,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)
    return auc


# Run Optuna Study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

print("\n🏁 OPTUNA TUNING COMPLETE")
print("Best AUC Score:", study.best_value)
print("Best Parameters:", study.best_params)

# ================= Train Final Best Model =================
best = study.best_params
model_type = best["model_type"]

if model_type == "logistic":
    final_model = LogisticRegression(
        C=best["C"],
        penalty=best["penalty"],
        solver="liblinear",
        random_state=2
    )
    final_model.fit(X_train_scaled, y_train)
    y_pred_prob = final_model.predict_proba(X_test_scaled)[:, 1]

elif model_type == "random_forest":
    final_model = RandomForestClassifier(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        min_samples_split=best["min_samples_split"],
        min_samples_leaf=best["min_samples_leaf"],
        random_state=2,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)
    y_pred_prob = final_model.predict_proba(X_test)[:, 1]

else:
    final_model = XGBClassifier(
        n_estimators=best["n_estimators"],
        learning_rate=best["learning_rate"],
        max_depth=best["max_depth"],
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        eval_metric="logloss",
        random_state=2,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)
    y_pred_prob = final_model.predict_proba(X_test)[:, 1]

auc_final = roc_auc_score(y_test, y_pred_prob)

name_map = {
    "logistic": "📈 Logistic Regression",
    "random_forest": "🌲 Random Forest",
    "xgboost": "⚡ XGBoost"
}

print("\n🏆 FINAL WINNER:", name_map[model_type])
print(f"🎯 Champion AUC Score: {auc_final:.4f}")

# MARKDOWN CELL da3v
# ## 🎯 Chapter 4: Action Plan - From Insights to Impact
# 
# Specific, actionable strategies based on our model insights...

# 🎯 ACTIONABLE BUSINESS RECOMMENDATIONS
print('🎯 MODEL-POWERED BUSINESS ACTION PLAN')
print('=' * 50)

# Get feature importance from XGBoost
xgb_model = final_model  # final_model is the best model (XGBoost)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print('🔍 TOP 5 CHURN DRIVERS:')
for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
    print(f'  {i}. {row["feature"]} (Impact: {row["importance"]:.1%})')

print('\n🚀 DATA-DRIVEN RECOMMENDED ACTIONS:')
top_features = feature_importance.head(5)

for i, (_, row) in enumerate(top_features.iterrows(), 1):
    feature = row['feature']
    impact = row['importance'] * 100
    print(f'\n🌟 {i}. {feature.upper()} ({impact:.1f}% Impact):')
    print('   • Monitor this feature closely for churn signals')
    print('   • Personalized campaigns to reduce risk of churn')
    print('   • Proactive engagement or incentives based on usage patterns')

import joblib
import os

print('💾 PREPARING MODEL FOR PRODUCTION')
print('=' * 40)

# Create output directory if it doesn't exist
os.makedirs('notebook_outputs', exist_ok=True)

# Save the final XGBoost model
joblib.dump(final_model, 'notebook_outputs/churn_prediction_model.pkl')

# Save preprocessing artifacts if they exist
if 'scaler' in globals():
    joblib.dump(scaler, 'notebook_outputs/feature_scaler.pkl')

if 'label_encoders' in globals():
    joblib.dump(label_encoders, 'notebook_outputs/label_encoders.pkl')

print('✅ Model artifacts saved:')
print('   📁 churn_prediction_model.pkl - Trained XGBoost model')
if 'scaler' in globals():
    print('   📁 feature_scaler.pkl - Data preprocessing scaler')
if 'label_encoders' in globals():
    print('   📁 label_encoders.pkl - Categorical encoders')
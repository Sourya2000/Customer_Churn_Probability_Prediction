# Customer Churn Prediction - Local Setup Guide

For live application : https://customerchurnprobabilityprediction-mnjwvsggrpxcavwrzmzyf8.streamlit.app/

## If the app is not opening click - Get the app back and it will be resumed after 2 minutes

<img width="1354" height="611" alt="image" src="https://github.com/user-attachments/assets/6120284a-41e8-4174-8c79-2dc272a1a105" />


## 📋 Project Overview
A machine learning system that predicts customer churn risk for telecom companies. The project includes a trained XGBoost model on AWS Sagemaker and an interactive Streamlit web application for real-time predictions.

## 🏗️ Project Structure
```
customer-churn-prediction/
├── app.py                          # Streamlit web application
├── notebooks/                      # Jupyter notebooks (if any)
├── churn_prediction_model.pkl      # Trained XGBoost model (681 KB)
├── feature_scaler.pkl              # Feature scaler for preprocessing (2 KB)
├── label_encoders.pkl              # Label encoders for categorical variables (2 KB)
├── README.md                       # Documentation
└── requirements.txt                # Python dependencies
```

## 🚀 Quick Start Guide

### Step 1: Clone or Download the Project
Download all files to a local directory.

### Step 2: Install Python
Make sure you have Python 3.8 or higher installed:
```bash
python --version
```

### Step 3: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies
Make sure your `requirements.txt` contains:
```txt
﻿pandas
numpy
scikit-learn
xgboost
joblib
streamlit
optuna
fpdf
```

Then install:
```bash
pip install -r requirements.txt
```

### Step 5: Run the Application
```bash
streamlit run app.py
```

This will open your web browser automatically at `http://localhost:8501`

## 🎯 Using the Application

### 1. Input Customer Information
Use the sidebar to enter customer details:
- **Customer Demographics**: State, Account Length, Area Code
- **Service Plans**: International Plan, VMail Plan
- **Usage Statistics**: Day/Eve/Night minutes and calls
- **Customer Service**: Number of support calls

### 2. Get Predictions
Click **"Analyze Customer"** to:
- Get churn risk classification (High/Low)
- View churn probability percentage
- See actionable insights based on key features

### 3. Understand Results
- **Low Risk (✅)**: Customer is likely to stay
- **High Risk (🚨)**: Customer is likely to leave
- **Probability**: Percentage chance of churn
- **Insights**: Specific risk factors identified

## 🔧 Troubleshooting

### Common Issues:

#### 1. "Module not found" errors
```bash
# Make sure all dependencies are installed
pip install --upgrade -r requirements.txt
```

#### 2. "File not found" errors
Ensure all .pkl files are in the same directory as `app.py`:
- `churn_prediction_model.pkl`
- `feature_scaler.pkl`
- `label_encoders.pkl`

#### 3. Streamlit won't start
```bash
# Check if Streamlit is installed
pip show streamlit

# Try running with explicit port
streamlit run app.py --server.port 8501
```

#### 4. Model loading errors
If the .pkl files are corrupted or incompatible:
- Delete the .pkl files
- Run the training script (if available)
- Or contact the project maintainer for fresh model files

### System Requirements:
- **RAM**: Minimum 4GB, 8GB recommended
- **Storage**: 100MB free space
- **Browser**: Chrome, Firefox, or Edge (latest versions)

## 📊 Model Details

### What's Included:
1. **churn_prediction_model.pkl** - XGBoost classifier trained on customer data
2. **feature_scaler.pkl** - StandardScaler for normalizing numerical features
3. **label_encoders.pkl** - Encoders for categorical variables (State, Plans)

### Features Used for Prediction:
- Account information (length, area code)
- Service plan details
- Usage patterns across day/evening/night
- International usage
- Customer service interactions
- Engineered features (total charges, average cost per minute)

## 🔄 Updating the Model

To retrain the model with new data:

1. **Prepare new data** in the same format as the training data
2. **Run the training script** (if provided)
3. **Replace the .pkl files** with newly trained versions
4. **Restart the Streamlit app**

## 📝 Notes for Developers

### File Locations:
- The app expects all .pkl files in the same directory
- Model files are already trained and ready to use
- No internet connection required after installation

### Customization:
- Modify `app.py` to change the UI/UX
- Update feature calculations in the prediction logic
- Add new visualizations to the results display

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify file locations and permissions
3. Ensure Python and all dependencies are properly installed
4. Contact the project maintainer for model-specific issues

## 📄 License
This project is provided for demonstration purposes. Please check with the project owner for licensing details.

---

**Enjoy predicting customer churn!** 🎯

*Last Updated: January 2026*



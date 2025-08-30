# 🏋️‍♂️ Calorie Expenditure Prediction Model

A **Machine Learning regression model** that accurately predicts calorie expenditure based on fitness and health parameters.  
Achieved **99.6% accuracy** using **LGBMRegressor** with hyperparameter tuning and cross-validation.

---

## 🚀 Features
- Predicts calorie expenditure with **99.6% test accuracy**
- Built with **LightGBM (LGBMRegressor)** for speed and performance
- Hyperparameter tuning with GridSearchCV
- Evaluated with multiple metrics:
  - ✅ R² Score (Train): **0.9964**
  - ✅ R² Score (Test): **0.9962**
  - ✅ Cross-Validation Score: **0.9963**
  - ✅ Low RMSE and MAE
- Clean, modular Python code

---

## 📂 Dataset
- Large dataset containing fitness-related attributes (e.g., heart rate, duration, body measurements)
- Preprocessed: scaling, feature engineering, null handling

---

## 📊 Model Workflow
1. **Data Preprocessing**
   - Missing value treatment
   - Feature selection
2. **Model Training**
   - LGBMRegressor with tuned hyperparameters:
     ```python
     {'num_leaves': 50, 'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1}
     ```
3. **Evaluation**
   - Train R²: **0.9964**
   - Test R²: **0.9962**
4. **Predictions**
   - Highly accurate calorie burn predictions for new data

---

## 🔧 Tech Stack
- Python 3
- Pandas, NumPy
- LightGBM
- Scikit-learn
- Matplotlib & Seaborn

📈 Results
Metric	Score
R² (Train)  	0.9964
R² (Test)	    0.9962
Cross-Val Score	0.9963
💡 Future Improvements

        Deploy as a web app (Streamlit/FastAPI)

        Add deep learning model for comparison

        Build fitness recommendation system

🤝 Contributing

Pull requests are welcome. Please fork this repo and submit PRs for suggestions.

📧 Contact
-Manvith


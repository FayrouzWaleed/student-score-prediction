# Student Score Prediction

Predicts students' exam scores from their study hours using linear regression.

## Dataset
[Student Performance Factors (Kaggle)](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
Download `StudentPerformanceFactors.csv` and place it in `data/`.

## What it does
1. Cleans the data (duplicates, missing values, invalid scores)
2. Basic visualization (distributions + hours vs score)
3. Train/test split (80/20)
4. Linear regression on `Hours_Studied`
5. Plots predictions and reports MAE, RMSE, R2

**Bonus**
- Polynomial regression (degree 1, 2, 3) compared with linear
- Different feature combinations (adding Sleep_Hours, Attendance, Previous_Scores)

## Run
```bash
pip install -r requirements.txt
python main.py
```
Plots are saved to `outputs/`; metrics are printed to the console.

## Tools
Python, Pandas, Matplotlib, Scikit-learn

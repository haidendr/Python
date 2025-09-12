# 🏀 NCAA Men's Basketball Data Analysis

This project uses Python to perform statistical and machine learning analysis on NCAA Men's Basketball data. The dataset comes from [Kaggle](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset?select=cbb.csv) and includes team performance metrics, conference affiliations, and tournament seedings.

## 📦 Dataset

- **Source**: Kaggle — [College Basketball Dataset](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset?select=cbb.csv)
- **File**: `cbb.csv`
- **Features**: Team name, conference, seed, wins, games played, offensive/defensive stats, and more.

## 🧪 Project Goals

- Perform exploratory data analysis (EDA) on NCAA teams and conferences
- Visualize team performance using bar charts
- Build predictive models using:
  - Linear Regression
  - Random Forest Regression
- Evaluate model performance using:
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² Score
  - Out-of-Bag (OOB) Score

## 🛠️ Technologies Used

- Python
- pandas, numpy — data manipulation
- matplotlib, seaborn — visualization
- scikit-learn — machine learning models and metrics

## 📊 Analysis Highlights

### Conference Breakdown
- Created separate DataFrames for each NCAA conference
- Visualized SEC team wins using a bar chart

### Linear Regression
- Imputed missing seed values with median
- Modeled relationship between seed and number of wins
- Visualized regression line and scatter plot
- Evaluated with MAE, MSE, RMSE, R²

### Random Forest Regression
- Dropped missing values for cleaner modeling
- Used seed and wins as features
- Evaluated with OOB Score, MAE, MSE, R²

  ## 📈 Sample Output
- Bar chart of SEC team wins
- Regression plot of seed vs. wins
- Model performance metrics printed to console

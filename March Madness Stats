# Using Python to perform statistical analysis on NCAA Men's Basketball data obtained from Kaggle.com
# dataset source: http://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset?select=cbb.csv
# keywords: forecast, prediction, exploratory analysis, linear regression, random forest, bar charts, imputation, pandas, numpy, matplotlib, seaborn, sklearn
# performance metrics used: MAE, MSE, RMSE, R-squared, OOB Score

# import pandas with 'pd' alias
import pandas as pd

# use read_csv to read kaggle.com basketball data set and store in 'cbb' dataframe
cbb = pd.read_csv('C:/Users/.../cbb.csv') 

# creating data frames of conferences
a10 = cbb[cbb["CONF"]=="A10"]     # Atlantic 10
acc = cbb[cbb["CONF"]=="ACC"]    # ACC conference
ae = cbb[cbb["CONF"]=="AE"]     # American East
aac = cbb[cbb["CONF"]=="Amer"]     # AAC
asun = cbb[cbb["CONF"]=="ASun"]     # Atlantic Sun
b10 = cbb[cbb["CONF"]=="B10"]     # Big 10
b12 = cbb[cbb["CONF"]=="B12"]
be = cbb[cbb["CONF"]=="BE"]       # Big East
bsky = cbb[cbb["CONF"]=="BSky"]     # Big Sky
bsth = cbb[cbb["CONF"]=="BSth"]     # Big South
bw = cbb[cbb["CONF"]=="BW"]     # Big West
caa = cbb[cbb["CONF"]=="CAA"]     # Big West
pac12 = cbb[cbb["CONF"]=="P12"]  # PAC12 conference
sec = cbb[cbb["CONF"]=="SEC"]

# creating a bar chart to show the team that won the most games
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects

wins_per_team = (
    sec[sec["W"]>0]
    .groupby("TEAM")["W"]
    .sum()
    .reset_index()
)

wins_per_team.head()
winners = wins_per_team.sort_values("W", ascending=False)

X = winners["TEAM"]
Y = winners["W"]

fig = plt.figure(figsize=(6,2), dpi = 150)
ax = plt.subplot(111)

ax.bar(X, Y, color = "#0033A0")

ax.tick_params(axis = "x", rotation = 90)
ax.set_title('SEC Conference')
ax.set_xlabel('Teams')
ax.set_ylabel('Wins')
plt.show()



# linear regression using seed and wins
# import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# import data
cbb = pd.read_csv('C:/Users...cbb.csv')

# looking at columns for missing data
cbb.isna().sum()

# imputation - fill nulls with median
impute_value = cbb['SEED'].median()
cbb['SEED'] = cbb['SEED'].fillna(impute_value)

# using the train test split function
#data = pd.DataFrame(x_train, columns=['SEED', 'G'])
#data.head()

X = np.array(cbb['SEED']).reshape(-1,1)
y = np.array(cbb['W']).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.60)
#x_train.head()

regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))

# exploring results
y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color = 'b')
plt.plot(X_test, y_pred, color = 'k')
plt.xlabel("Seed")
plt.ylabel("Number of Wins")
plt.title("Linear Regression Results")
plt.show()

# performance measures for linear regression
from sklearn.metrics import mean_absolute_error,mean_squared_error
 
mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
#squared True returns MSE value, False returns RMSE value.
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
 
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)



# random forest model
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('C:/Users...cbb.csv')
df.sample(5, random_state=42)

# remove missing data instead of imputation
df = df.dropna()
#df.info()

# separating the features (x) and the labels (y)
X = df[['SEED', 'W']]
y = df['W']

# training our random forest mdoel
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

# FITTING RANDOM FOREST Regression TO THE DATASET
#rf_model = RandomForestClassifier(n_estimators=10, max_features="sqrt", random_state=0, oob_score=True)
regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
 
# Fit the regressor with x and y data
regressor.fit(X_train, y_train)

# Evaluating the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
# Access the OOB Score
oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')
 
# Making predictions on the same data or new data
predictions = regressor.predict(X)
 
# Evaluating the model
mae = mean_absolute_error(y, predictions)
print(f'Mean Absolute Error: {mae}')

mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error: {mse}')

#rmse = root_mean_squared_error(y, predictions)
#print(f'Mean Squared Error: {rmse}')

r2 = r2_score(y, predictions)
print(f'R-squared: {r2}')

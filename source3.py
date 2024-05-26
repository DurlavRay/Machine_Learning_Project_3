import pandas as pd
import seaborn as sns
df = pd.read_csv("IPL.csv")
df
from sklearn.model_selection import train_test_split
df.isnull().sum()
df
x = df[['Run after 6th over', 'Wicket after 6th over', 'Run after 16th over', 'Wicket after 16th over']]
y = df.Score
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15, random_state=42)
from ctypes import LibraryLoader
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
model.score(x,y)
from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor(criterion = 'friedman_mse', random_state=0)
model2.fit(x_train,y_train)
model2.score(x,y)
import pickle
filename = 'ipl_regression.sav'
pickle.dump(model, open(filename, 'wb'))
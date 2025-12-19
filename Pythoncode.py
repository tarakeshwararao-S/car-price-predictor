#importing all the useful modules for the project
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor,LinearRegression
df=pd.read_csv('/content/CarPrice_Assignment.csv')
#dropped them because they are not contibuting in predicting the output
df=df.drop(columns=["car_ID","CarName"])
df
#no null values
#finding the columns with categorical values and converting them into numerical values using one-hot Encoding
cat_cols=df.select_dtypes(include='object').columns
df=pd.get_dummies(df,columns=cat_cols,drop_first=True)
#splitting the data for training and testing purpose
input=df.drop(columns=["price"])
output=df["price"]
x_train,x_test,y_train,y_test=train_test_split(input,output,random_state=42,test_size=0.2)
#performing feature scaling for input values
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
#now fitting our model using sgdregressor which is linear regression model
#uses gradient descent to calculate the best parameters
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
import joblib

# save trained model
joblib.dump(model, "car_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(input.columns, "columns.pkl")
print("All files saved successfully!")

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

#DATASET
def load_data():
  df=pd.read_csv('cars24-car-price-clean2.csv')
  
  X=df[['year', 'km_driven', 'mileage']].values
  y=df['selling_price'].values
  return X,y

#SPLITTING THE DATASET
def split_data(X,y):
  return train_test_split(X,y,test_size=0.2,random_state=42)

#Train the model
def train_model(X_train,y_train):
  model=LinearRegression()
  model.fit(X_train,y_train)
  return model

#Prediction
def predict(model,X_test,y_test):
  y_pred=model.predict(X_test)
  comparision_df=pd.DataFrame({'Actual':y_test,'Prediction':y_pred})
  print('Comparison DataFrame:')
  print(comparision_df)
  return y_pred

#Evalution
def evalution(y_test,y_pred):
  print('Evalutuon:')
  print("MSE:",mean_squared_error(y_test,y_pred))
  print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred)))
  print("R2Score:",r2_score(y_test,y_pred))

#Plotting
def plot(X_test,y_test,y_pred):
  plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--',label='Ideal')
  plt.xlabel('Actual_Selling_Price')
  plt.ylabel('Predicted_Selling_Price')
  plt.title('Car Selling Price Prediction')
  plt.legend()
  plt.grid()
  plt.show()

if __name__=='__main__':
    X,y=load_data()
    X_train,X_test,y_train,y_test=split_data(X,y)
    model=train_model(X_train,y_train)
    y_pred=predict(model,X_test,y_test)
    evalution(y_test,y_pred)
    plot(X_test,y_test,y_pred)

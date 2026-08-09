import pandas as pd 
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

#DATASET
def load_data():
    df=pd.read_csv('train.csv')
    X=df[['trip_duration']]
    y=df['distance_traveled']
    return X,y

#SPLITTING THE DATASET
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

#TRAIN THE MODEL
def train_model(X_train, y_train):
    model=LinearRegression()
    model.fit(X_train, y_train)
    return model

#PREDICTION
def predict(model,X_test,y_test):
    y_pred=model.predict(X_test)
    comparsion_df=pd.DataFrame({'Actual':y_test,'Prdediction':y_pred})
    print('sample Prediction:')
    print(comparsion_df)
    print('           ')
    return y_pred

#EVALUATION
def evalution(y_test,y_pred):
    print('Evaluation:')
    print("MSE",mean_squared_error(y_test,y_pred))
    print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))
    print("R2_score",r2_score(y_test,y_pred))

 #PLOTTING
def plot():
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test,y_pred,color='green', label='Predicted')
    plt.xlabel('Trip Duration')
    plt.ylabel('Distance Traveled')
    plt.title('Trip Duration vs Distance Traveled')
    plt.legend()
    plt.show()

if __name__=='__main__':
    X,y=load_data()
    X_train, X_test, y_train, y_test=split_data(X, y)
    model=train_model(X_train, y_train)
    y_pred=predict(model,X_test,y_test)
    evalution(y_test,y_pred)
    plot()
    

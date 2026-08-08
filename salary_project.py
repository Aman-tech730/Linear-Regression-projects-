import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

#DATASET
def load_data():
    df = pd.read_csv('Salary_dataset.csv')
    print(df.columns)  
    X = df[['Ex']]   
    y = df['Salary']
    return X, y

#Split The Datasets Into Training And Test Datasets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

#Train The Model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

#PREDICTION
def predict(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print('Sample prediction:')
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(comparison_df)
    print('           ')
    return y_pred

#EVALUATION
def evaluation(y_test, y_pred):
    print('           ')
    print('Evaluation:')
    print('MSE:', mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R2 Score:', r2_score(y_test, y_pred))

#PLOTTING 
def plot(X_test, y_test, y_pred):
    plt.scatter(X_test, y_test, label='Actual')
    plt.plot(X_test, y_pred, color='red', label='Predicted')
    plt.legend()
    plt.grid()
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Salary Prediction')
    plt.show()

if __name__ == '__main__':
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    y_pred = predict(model, X_test, y_test)
    evaluation(y_test, y_pred)
    plot(X_test, y_test, y_pred)

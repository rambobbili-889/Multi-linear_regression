import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error
import sys


class MULTI_LINEAR_REGRESSION:
    def __init__(self, location):
        try:
            self.df = pd.read_csv(location)  # Here we are reading the data file
            self.df = self.df.drop(['Customer Name', 'Customer e-mail', 'Country'], axis=1)  # Here we are removing the columns which we don't require
            self.X = self.df.iloc[:, :-1]  # we are give the independent columns to the X variable
            self.Y = self.df.iloc[:, -1]   # we are give the dependent column to the Y variable
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=99)  # Dividing the into train and test using train_test_split
            self.data_train = pd.DataFrame()  # creating a dataframe
            self.data_train = self.x_train.copy()  # Giving the x train value to the dataframe
            self.data_train['y_train_values'] = self.y_train  # giving y train values to the dataset
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def training_data(self):
        try:
            self.reg = LinearRegression()  # giving object to the class
            self.reg.fit(self.x_train,self.y_train) # training the model using Linear_model
            self.y_train_predict = self.reg.predict(self.x_train)  # Here the y train predict values will be generated
            self.data_train['y_train_predict'] = self.y_train_predict
            print(f'the training data is : \n {self.data_train}')
            print(f'The accuracy of the training data  is : {r2_score(self.y_train,self.y_train_predict)}')  # Find the accuracy of the training data
            print(f'The loss factor using the Mean Squared Error is :{mean_squared_error(self.y_train,self.y_train_predict)}')
            print(f'The loss factor using the root_mean_squared_error is :{root_mean_squared_error(self.y_train, self.y_train_predict)}')
            print(f'The loss factor using the  mean_absolute_error is :{mean_absolute_error(self.y_train, self.y_train_predict)}')
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def testing_data(self):
        try:
            self.data_test = pd.DataFrame()  # creating the dataframe for test for better understanding
            self.data_test = self.x_test.copy()  # Giving the x test value to the dataframe
            self.data_test['y_test_values'] = self.y_test  # giving y test values to the dataset
            self.y_test_predict = self.reg.predict(self.x_test)
            self.data_test['y_test_predict_values'] = self.y_test_predict
            print(f'the testing data is : \n{self.data_test}')
            print(f'The accuracy of the testing data  is : {r2_score(self.y_test, self.y_test_predict)}')  # Find the accuracy of the training data
            print(f'The loss factor of the testing data using the Mean Squared Error is :{mean_squared_error(self.y_test, self.y_test_predict)}')
            print(f'The loss factor of the testing data  using the root_mean_squared_error is :{root_mean_squared_error(self.y_test, self.y_test_predict)}')
            print(f'The loss factor of the testing data using the  mean_absolute_error is :{mean_absolute_error(self.y_test, self.y_test_predict)}')
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def manual_finding_meansquared_error(self):
        try:
            self.count = 0
            for i in self.data_train.index:
                self.count = self.count + (self.data_train['y_train_values'][i] - self.data_train['y_train_predict'][i])**2
            print(f'The Accuracy calculated normally is : {self.count/len(self.y_train)}')
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)


if __name__ == "__main__":
    try:
        obj = MULTI_LINEAR_REGRESSION('C:\\Users\\Bunny\\PycharmProjects\\Machine_Learning_algorithms\\Car_Purchasing_Data.csv')
        obj.training_data()
        obj.testing_data()
        obj.manual_finding_meansquared_error()
    except Exception as e :
        error_type, error_message, error_line_no = sys.exc_info()
        print(error_type, error_message, error_line_no.tb_lineno)
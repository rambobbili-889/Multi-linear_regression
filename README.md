# Multi-linear_regression
Project Title: Multi Linear Regression on Car Purchasing Data
Overview
This project implements a Multi-Linear Regression model using the sklearn library in Python. The aim is to predict a customer's car purchase amount based on multiple factors provided in the dataset. The program reads data from a CSV file, splits the data into training and testing sets, trains the model, and evaluates its performance using metrics such as R-squared, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

Prerequisites
Make sure you have the following libraries installed in your environment:

bash
Copy code
pip install numpy pandas scikit-learn
Dataset
The dataset used is the Car_Purchasing_Data.csv, which contains customer-related data. The data fields include:

Customer Name
Customer e-mail
Country
Various other numeric columns for demographic and financial data
Note: The columns Customer Name, Customer e-mail, and Country are dropped during preprocessing as they are irrelevant to the prediction model.

File Structure
plaintext
Copy code
MULTI_LINEAR_REGRESSION.py
Car_Purchasing_Data.csv
README.md
Code Explanation
1. Class Initialization
python
Copy code
class MULTI_LINEAR_REGRESSION:
    def __init__(self, location):
        # Reads the CSV file from the provided location
        self.df = pd.read_csv(location)
        
        # Dropping irrelevant columns: 'Customer Name', 'Customer e-mail', 'Country'
        self.df = self.df.drop(['Customer Name', 'Customer e-mail', 'Country'], axis=1)
        
        # Independent variables (X) and dependent variable (Y)
        self.X = self.df.iloc[:, :-1]  # X includes all columns except the last one
        self.Y = self.df.iloc[:, -1]   # Y is the last column, which is the target column (car purchase amount)
        
        # Splitting the dataset into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=99)
        
        # Creating a copy of the training data for analysis
        self.data_train = self.x_train.copy()
        self.data_train['y_train_values'] = self.y_train
Location: The CSV file's path is passed to the class for data reading.
Data Preprocessing: Unnecessary columns are removed to avoid irrelevant data affecting the regression model.
Train-Test Split: The dataset is split into training and testing sets (80% train, 20% test).
2. Training the Model
python
Copy code
def training_data(self):
    self.reg = LinearRegression()  # Initializing the Linear Regression model
    self.reg.fit(self.x_train, self.y_train)  # Training the model with training data
    
    # Predicting Y values for the training set
    self.y_train_predict = self.reg.predict(self.x_train)
    
    # Adding predicted values to the training dataset for comparison
    self.data_train['y_train_predict'] = self.y_train_predict
    
    # Printing the training data with actual and predicted values
    print(f'the training data is : \n {self.data_train}')
    
    # Printing performance metrics
    print(f'The accuracy of the training data is: {r2_score(self.y_train, self.y_train_predict)}')
    print(f'The loss factor using the Mean Squared Error is: {mean_squared_error(self.y_train, self.y_train_predict)}')
    print(f'The loss factor using the root_mean_squared_error is: {root_mean_squared_error(self.y_train, self.y_train_predict)}')
    print(f'The loss factor using the mean_absolute_error is: {mean_absolute_error(self.y_train, self.y_train_predict)}')
Model Training: A linear regression model is fitted using the training data.

Predictions: The model predicts the target variable for the training set.

Performance Metrics:

R-squared: Shows how well the model explains the variance in the target variable.

Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.

Root Mean Squared Error (RMSE): The square root of MSE.

Mean Absolute Error (MAE): The average absolute difference between actual and predicted values.

3. Testing the Model
python
Copy code
def testing_data(self):
    # Creating a copy of the test data
    self.data_test = self.x_test.copy()
    self.data_test['y_test_values'] = self.y_test
    
    # Predicting Y values for the testing set
    self.y_test_predict = self.reg.predict(self.x_test)
    
    # Adding predicted values to the test dataset for comparison
    self.data_test['y_test_predict_values'] = self.y_test_predict
    
    # Printing the testing data with actual and predicted values
    print(f'the testing data is : \n{self.data_test}')
    
    # Printing performance metrics for the test data
    print(f'The accuracy of the testing data is: {r2_score(self.y_test, self.y_test_predict)}')
    print(f'The loss factor of the testing data using the Mean Squared Error is: {mean_squared_error(self.y_test, self.y_test_predict)}')
    print(f'The loss factor of the testing data using the root_mean_squared_error is: {root_mean_squared_error(self.y_test, self.y_test_predict)}')
    print(f'The loss factor of the testing data using the mean_absolute_error is: {mean_absolute_error(self.y_test, self.y_test_predict)}')

Testing: Predictions are made on the test dataset using the trained model.

Evaluation: The same metrics used for training data are calculated and printed for the test data to assess model generalization.
Usage

Place the CSV dataset (Car_Purchasing_Data.csv) in the same directory as the Python script.

Run the script by executing:

bash

Copy code

python MULTI_LINEAR_REGRESSION.py

Key Features

Model Training: The model is trained using the training set split from the original dataset.

Model Evaluation: The model's performance is evaluated on both training and testing sets using multiple metrics.

Data Preprocessing: Unnecessary columns are automatically removed before applying the model.

Notes

The current code only focuses on removing specific columns (Customer Name, Customer e-mail, Country). If more columns are irrelevant, adjust the drop() function accordingly.

The dataset is split randomly, but the random state ensures reproducibility.


Output 
Training Data Output:
python
vbnet
Copy code
The accuracy of the training data  is : 0.9994907891819212

The loss factor using the Mean Squared Error is :57280.69040331835

The loss factor using the root_mean_squared_error is :239.3338471744403

The loss factor using the  mean_absolute_error is :207.31073537356392

python
vbnet
Copy code
The accuracy of the testing data  is : 0.9995270396830659

The loss factor of the testing data using the Mean Squared Error is :60531.52803897554

The loss factor of the testing data  using the root_mean_squared_error is :246.03155903049418

The loss factor of the testing data using the  mean_absolute_error is :213.03617288115566


Conclusion
This project demonstrates a simple implementation of Multi-Linear Regression using the scikit-learn library. It provides a solid foundation for analyzing customer-related datasets and predicting target variables based on multiple factors. The results include detailed metrics for both training and testing data, offering insights into the model's accuracy and error rates.

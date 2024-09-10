Multi-Linear Regression on Car Purchasing Data
Overview
This project implements a Multi-Linear Regression model using the scikit-learn library in Python. The objective is to predict the amount a customer will spend on purchasing a car based on various factors provided in the dataset.

The program follows these steps:

Reads the data from a CSV file.
Preprocesses the data by removing unnecessary columns.
Splits the dataset into training and testing sets.
Trains a linear regression model on the training data.
Evaluates the model's performance using key metrics such as:
R-squared
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
Prerequisites
Ensure that the following Python libraries are installed:

bash
Copy code
pip install numpy pandas scikit-learn
Dataset
The dataset used in this project is Car_Purchasing_Data.csv, which contains various fields related to customers. Here are the fields:

Customer Name (Dropped during preprocessing)
Customer e-mail (Dropped during preprocessing)
Country (Dropped during preprocessing)
Other numerical columns related to customer demographics and financial data
Preprocessing
Unnecessary columns, such as Customer Name, Customer e-mail, and Country, are removed during preprocessing to focus on the relevant numerical data for prediction.

File Structure
plaintext
Copy code
MULTI_LINEAR_REGRESSION.py
Car_Purchasing_Data.csv
README.md
Code Explanation
1. Class Initialization
The MULTI_LINEAR_REGRESSION class handles the entire regression process. Here's how it's set up:

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
Key Points:

Location: The CSV file's path is passed to the class for data reading.
Data Preprocessing: Columns that are irrelevant to the prediction are dropped.
Train-Test Split: The dataset is split into training (80%) and testing (20%) sets.
2. Training the Model
The training_data method trains the linear regression model and evaluates its performance on the training data.

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
    print(f'The loss factor using the root_mean_squared_error is: {np.sqrt(mean_squared_error(self.y_train, self.y_train_predict))}')
    print(f'The loss factor using the mean_absolute_error is: {mean_absolute_error(self.y_train, self.y_train_predict)}')
Key Features:

Model Training: The model is trained using the LinearRegression class from scikit-learn.

Predictions: The model predicts car purchase amounts for the training set.

Performance Metrics:
R-squared: Indicates how well the model explains the variance in the dependent variable.

Mean Squared Error (MSE): Average squared difference between actual and predicted values.

Root Mean Squared Error (RMSE): Square root of MSE.

Mean Absolute Error (MAE): Average absolute difference between actual and predicted values.

3. Testing the Model
4. 
The testing_data method evaluates the model's performance on the testing data:

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
    print(f'The loss factor of the testing data using the root_mean_squared_error is: {np.sqrt(mean_squared_error(self.y_test, self.y_test_predict))}')
    print(f'The loss factor of the testing data using the mean_absolute_error is: {mean_absolute_error(self.y_test, self.y_test_predict)}')
Testing:

The model is evaluated on the test dataset to assess its generalization performance.

The same metrics used for the training data are printed for the testing data.

Usage

To run the project, place the Car_Purchasing_Data.csv in the same directory as MULTI_LINEAR_REGRESSION.py, then execute:

bash

Copy code
python MULTI_LINEAR_REGRESSION.py
Key Features

Model Training: The model is trained on 80% of the dataset.

Model Evaluation: The model is evaluated on both training and testing datasets.

Preprocessing: Automatically removes irrelevant columns.

Notes

The current code drops Customer Name, Customer e-mail, and Country. If other columns need to be excluded, modify the drop() function.

The dataset split uses a random_state to ensure reproducibility.

Sample Output

Training Data Output:

plaintext

Copy code
The accuracy of the training data is : 0.9994907891819212
The loss factor using the Mean Squared Error is : 57280.69040331835
The loss factor using the root_mean_squared_error is : 239.3338471744403
The loss factor using the mean_absolute_error is : 207.31073537356392

Testing Data Output:

plaintext

Copy code

The accuracy of the testing data is : 0.9995270396830659
The loss factor of the testing data using the Mean Squared Error is : 60531.52803897554
The loss factor using the root_mean_squared_error is : 246.03155903049418
The loss factor using the mean_absolute_error is : 213.03617288115566

Conclusion
This project demonstrates how to implement Multi-Linear Regression using Python and scikit-learn. It provides insight into predicting customer car purchasing amounts based on several factors, and uses key metrics to evaluate model performance for both training and testing datasets.

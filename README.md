# machinelearning-az
Notes for Udemy course on Machine Learning A-Z



## Section: 1 - Welcome to the course!

### 1 -  Applications of Machine Learning 3:22
### 2 -  Why Machine Learning is the Future 6:37
### 3 -  Installing R and R Studio (MAC & Windows) 5:40
### 4 -  Installing Python and Anaconda (MAC & Windows) 7:31

* Download Anaconda
* https://www.continuum.io/downloads
* Anaconda is an IDE package on top of Python and Python packages
* Launch Spyder
* In the Window Panes you want Editor, Interactive Python console and Variabel Explorer with Help
* In editor  `> print("Hello World")`
* Highlight and press CTRL-Enter and see it appear in the interactive console.

### 5 -  BONUS: Meet your instructors

## Part 1: Data Preprocessing - Section: 2 0 / 11

### 6 - Welcome to Part 1 - Data Preprocessing 1:35

* We need ot start out with Data PreProcessing to get to the fun parts later
* This involves downloading a lot of datasets and processing them.

### 7 - Get the dataset 6:58

* Go to: https://www.superdatascience.com/machine-learning/
* Unzip both files
* Place Preprocsessing in Template folder structures
* First dataset is Data.csv - first 3 rows are the independent variables, last row is dependent

### 8 - Importing the Libraries 5:20

* We need to create a file for the Data Preprocessing Template - data_processing_template.py
* We need to import 3 basic libraries
* `import numpy as np`
* `import matplotlib.pyplot as plt` - to plot math charts, anytime you want to plot something in Python
* `import pandas as pd` - best library to import and manage datasets
* Highlight this cose and hit CTRL-Enter to execute to make sure it is in correctly.
* Note: in R you don't have to separately load the packages.

### 9 - Importing the Dataset 11:55

* `dataset = pd.read_csv('Data.csv')` - Add this to import the dataset
* In variable explorer you can see the dataset
* Change the salary from scientific notation: from `%.3g` to `%.0f`
* Let's start creating our matrix of features
* Add new code for the data:
```
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
```

* First, we take all the lines (left of first comma) and then all but the last line (all but last column, right of comma)
* Execute that line and type X in the console. This is our matrix of independent variables.
* Y is goign to be for the last column

### 10 - For Python learners, summary of Object-oriented programming: classes & objects 0:00

### 11 - Missing Data 15:57

* Now we are going to deal with missing data in the dataset.
* We're missing data in columns for Spain and German.
* One idea si to remove the line -- but we can't do that.
* Most common: take the mean of the columns/
* `from sklearn.preprocessing import Imputer`
* This imports a library impute which allows us to handle missing data
* Now we need to create an object
* `imputer = Imputer(missing_values = 'NaN')`
* We're switching out NaN - reason is if you look in "Variable Explorer" at Data.csv in DataFrame mode you will see NaN in missing blanks.
* Now we make a strategy for mean = `imputer = Imputer(missing_values = 'NaN',strategy = 'mean')`
* Now we set axis=0 for columns `imputer = Imputer(missing_values = 'NaN',strategy = 'mean', axis = 0)`
* `imputer = imputer.fit(X[:,1:3])` - we're taking the 1 and 2 rows but not 3 (1:3 means 1 and 2 but not 3)
* Run the impute part of the code
* in console: `X` and this should output all the rows
* (you may need to also input into console: `np.set_printoptions(threshold=100) `) if the rows are truncated.
* Check Data.cvs in Spreadsheet, get avg. salary `=AVERAGE(C1:C11)`
* Output: 63777.7777777778
* Note: for startegies you can also take the "median" and "most frequent" values

### 12 - Categorical Data 18:01

* The Country and Purchase columns are called Category columns (Germany/France/Spain Yes/No)
* We have to get the text out of the machine learning equations
* We need to encode the text into numbers.
* `from sklearn.preprocessing import LabelEncoder`
* Then we have to create an objects

```
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
```

* Run in console.
* Unfortunately at this point we have higher and lower numbers for each country which could make one seem greater than another.
* So instead we'll break them into 3 columns of 1 or 0
* To do this we need to import OneHotEncoder `from sklearn.preprocessing import LabelEncoder, OneHotEncoder`

7:38

* INFO: To get info on an object got to Help `sklearn.preprocessing.OneHotEncoder`
* Add in the following code:
```
#Encoding Category data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
```

* Run, Now check in Variable explorer - double-mouse-click x - you should see 3 columns prepended with 1s and 0s
* Next we'll take care of the purchased Column
* Copy paste this part:
```
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
```
* change to y

```
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

```

* Now check in Variable explorer - double-mouse-click y - you should see 1 columns with 1s and 0s

### 13 - Splitting the Dataset into the Training set and Test set 17:37

* We have to split the Dataset into a Training and a Test set.
* The test set with have slightly different data.
* The test set is used to test the perfromance of how well we trained the ML.
* We are testing the adpatation of the rules to a new set of data.
* We expect there should not be much difference in performance.
* It's very simple, takes 2 lines:
```
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

```

* These are our dependent and independent variables by each set: X_train, X_test, y_train, y_test
* in train_test_split we need to cite X, y which is the whole dataset. test_size 0.2 is 20%
* We have 10 observations in the train set, 2 in test set
* random_state is if you want random sampling.
* Select these lines and Run
* See in Variable explorer the new datasets
* Note that for X in train we have 10 observations and in test we have 2.

### 14 - Feature Scaling 15:36

* What is feature scaling and wy do we need to do it?
* The Euclidean Distance will be affected between the Age and Salary scale differences. (max and min for each column)
* We need to transform the variabels tot he ame scale.
* see graphic: 14-Standardization-Normalization
* import scaling library: `from sklearn.preprocessing import StandardScaler`
* Then we are going to fit_transform each the train dataset X,y - and only transform the test set
* Code:
```
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```
* Run, see result graphic: 14-X-Standardization
* This is all that is requried to preprocess data

### 15 - And here is our Data Preprocessing Template! 8:48

* We only include libraries we need.
* See: Preprocessing Template graphic
* For template we'll remove some of what qwe did so far
* REMOVE or COMMENT OUT - Taking care of missing data
* REMOVE or COMMENT OUT - Encoding Category data
* COMMENT OUT - Feature scaling
* Every time we start a machine learning model we will copy/paste this template
*

    Quiz 1: Data Preprocessing 0:00

## Section 4: Simple Linear Regression

We're going to handle this next:

* Simple Linear Regression
* Multiple Linear Regression
* Polynomial Regression
* Support Vector for Regression (SVR)
* Decision Tree Classification
* Random Forest Classification

### 17 - How to get the dataset 3:18

### 18 - Dataset + Business Problem Description 2:56

* Download dataset: https://www.superdatascience.com/machine-learning/

### 19 - Simple Linear Regression Intuition - Step 1 5:45

* Data: Simple Linear Regression/Salary_Data.csv
* What is the correlation between salary and years experience.
* What is the business value add? This is the model current and what should we apply?

### 20 - Simple Linear Regression Intuition - Step 2 3:09
### 21 - Simple Linear Regression in Python - Step 1 9:55

* Linear Regression: y = b(0) + b(1) * x
* Image: 21-Simple-Linear-Regression
* Image: 21-Simple-Linear-Regression-Dependent-Variable
* Image: 21-Simple-Linear-Regression-Independent-Variable
* Image: 21-Simple-Linear-Regression-Coefficient

Example:

* So we start with an x (Experience) and y (salary) axis
* So we plot Observations on the x and y axis
* Linear Regression: Salary = b(0) + b(1) * Experience
* Linear regression means the plotted line, slope proportion

* Image: 21-Simple-Linear-Regression-FULL-EXAMPLE.png

### 22 - Simple Linear Regression in Python - Step 2 8:19

* See ordinary least squares image. 22-Ordinary-Least-Squares.png
* 22-Ordinary-Least-Squares-2-difference.png - y-i (red) and y-i-hat (green)
* This is the difference between what is observed and the model
* Take the differnece of that and take the SUm of the squares... SUM(y - y^)^2 -> min
* So it takes the gaps and sums them, and takes the linbe that has the minimal sum of squares possible.
* See image: 22-Ordinary-Least-Squares-3-SUM.png
*
### 23 - Simple Linear Regression in Python - Step 1 6:43

* Setup  Simple Linear Regression script in Spyder
* First thing we need to do is use our Data Processing template (last file made, previous section) to get started. Copy paste
* Update csv to Salary_Data.csv and import into Variable Expolorer
* We have 30 observations (30 employees)
* We want to train and establish a correlation between experience and salary.
* We have to SPLIT the data out first.
* X is the matrix of features (dependent variable)
* Independent variable is the Years for experience
* Dependent variable is the salary

```
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
```

* X removes the last column
* y will be 1 becasue that is the independent variable column
* Run that code though X and you should get X with one column
* Run code for y and you should get y with one column

* At this point we have split the orginal dataset. Now, we have to split into a (1) Train and (2) Test Sets

* `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)`

* We want to test size to be less than a half -- let;s do 1/3 for a round 10 number (1/3 of 30)
* Execute this code. It divides data sets again. See img: 23-Train-Test-Sets.png
* We're using X_train and y_train to get the correlations and then we will use the result in the Test groups

* Next step is FEATURE SCALING & FITTING the algorithm to our Dataset

### 24 - Simple Linear Regression in Python - Step 2 14:50

* Feature Scaling we'll leave commented out for now.

* Our data has been preprocessed. Now we have to fit the algorithm.
* We need to import the Linear Regression class
`from sklearn.linear_model import LinearRegression`
* Out of this we are going to make an object that will be our Linear Regressor
* The Regressor object will use the fit fethod to fit to the data model.
```
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```
* Check help for info on the LinearRegression class.
* Now this code can be executed.
* Result:
```
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
Out[13]: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
```

* That is it for the most basic Linear Regression Machine Learning Model

* In the next section we'll use it to predict some new observations which will the test set observations.

### 25 - Simple Linear Regression in Python - Step 3 6:43

* First step: was to preprocess the data.
* Second step: create linear regression model
* Next we'll predict the Test set results
* We'll create a vector with the test set salaries called y_

`y_pred = regressor.predict(X_test)`

* y_pred is always the vector of predictions for the Dependent variable
* `predict` is a method of the LinearRegression class.
* check help for info about predict
* Execute the code
* New y_pred row - See result: 25-1-Result-of-y_pred.png
* Open y_pred and y_test datasets
* What is the difference?
* y_test is the real salaries observed
* y_pred is the predicted salaries
* Compare the two datasets - test and predicted - they are not perfect, some are close some aren't.

### 26 - Simple Linear Regression in Python - Step 4
### 25 - Simple Linear Regression in R - Step 3 4:40
### 26 - Simple Linear Regression in R - Step 4 5:58
### 27 - Simple Linear Regression in R - Step 3 3:38
### 28 - Simple Linear Regression in R - Step 4 15:55
    Quiz 2: Simple Linear Regression 0:00



### 31 - How to get the dataset 3:18

* General instructions about getting dataset.

### 32 - Dataset + Business Problem Description 3:44

* venture capital dataset
* 5 columns
* 50 companies
* View CSV - 50_Startups.csv
* Fields: R&D Spend, Administration, Marketing Spend, State, Profit
* We need to create a model to decide which types of companies are best to invest in based on Profit.
* Dependent variable (DV): Profit. Other variables are independent variables (IV).
* They need to find out which companies do better on various factors.

### 33 - Multiple Linear Regression Intuition - Step 1 1:02

* see image:  33-Multiple-Regression-Formula.jpg
* Multiple Regresion Formula: y = b(0) + b(1)\*x(1) + b(2)\*x(2) etc.
* See image on full desciption of formula: 33-2-Multiple-Regression-Formula--FULL-Descriptions.png

### 34 - Multiple Linear Regression Intuition - Step 2 1:00

* Quick heads up -- there is a Caveat about Linear Regressions.
* Linear Regressions have assumptions.
* See image: 34-1-Linear-Regressions-Assumptions.jpg
* Linearity, Homoscedasticity, Multivariate normality, Independence of errors, Lack of multicollinerity
* Always make sure your Assumptions are correct when buuilding a Linear Regression.

### 35 - Multiple Linear Regression Intuition - Step 3 7:21





### 36 - Multiple Linear Regression Intuition - Step 4 2:10



### 37 - Multiple Linear Regression Intuition - Step 5 15:41



### 38 - Multiple Linear Regression in Python - Step 1 15:57



### 39 - Multiple Linear Regression in Python - Step 2 2:56



### 40 - Multiple Linear Regression in Python - Step 3 5:28



### 41 - Multiple Linear Regression in Python - Backward Elimination - Preparation 13:14



### 42 - Multiple Linear Regression in Python - Backward Elimination - HOMEWORK ! 12:40



### 43 - Multiple Linear Regression in Python - Backward Elimination - Homework Solution 9:10




    .

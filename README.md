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
### 15 - And here is our Data Preprocessing Template! 8:48
    Quiz 1: Data Preprocessing 0:00



.

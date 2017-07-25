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
### 12 - Categorical Data 18:01
### 13 - Splitting the Dataset into the Training set and Test set 17:37
### 14 - Feature Scaling 15:36
### 15 - And here is our Data Preprocessing Template! 8:48
    Quiz 1: Data Preprocessing 0:00



.

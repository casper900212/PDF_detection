import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import csv
import joblib


class dtdata:
    def __init__(self,filename):
        self.filename = filename
        self.original_label = 2
        self.predict_label = 2
        self.js = 0 

# read csv file
df_train = pd.read_csv("./traindata.csv",encoding='utf-8')

# train and return the predict result

# fill the missing value
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
X_train = imp.fit_transform(df_train.drop(['Malware', 'filename', 'header', 'version'], axis=1))
y_train = df_train['Malware']

# build decision tree model
clf = DecisionTreeClassifier(max_depth=35, max_features=5, min_samples_leaf=1, min_samples_split=5, random_state=42)

# train model
clf.fit(X_train, y_train)

joblib.dump(clf,'decision_tree_pdf.pkl')

    


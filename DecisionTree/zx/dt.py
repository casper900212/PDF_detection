import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import csv


class dtdata:
    def __init__(self,filename,original_label):
        self.filename = filename
        self.original_label = int(original_label)
        self.predict_label = 2
        self.js = 0 

# read test data    
def read_test(filename):
    return pd.read_csv(filename,encoding='utf-8')

# read csv file
df_train = pd.read_csv("traindata.csv",encoding='utf-8')
df_test = pd.read_csv("testdata.csv",encoding='utf-8')

# data preprocessing
def data_preprocessing(inputdata,inputfilename):
    testlist = []
    with open(inputfilename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row["filename"]
            orglabel = row["Malware"]
            testlist.append(dtdata(filename,orglabel))
    return testlist

# train and return the predict result
def DecisionTree(testdata):
    # fill the missing value
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    X_train = imp.fit_transform(df_train.drop(['Malware', 'filename', 'header', 'version'], axis=1))
    y_train = df_train['Malware']

    X_test = imp.transform(df_test.drop(['Malware', 'filename', 'header', 'version'], axis=1))
    y_test = df_test['Malware']

    # build decision tree model
    clf = DecisionTreeClassifier(max_depth=35, max_features=5, min_samples_leaf=1, min_samples_split=5, random_state=42)

    # train model
    clf.fit(X_train, y_train)

    # predict test data
    y_pred = clf.predict(X_test)
    
    index = 0
    for x in y_pred:
        testdata[index].predict_label = x
        index += 1
    return testdata

# calculate accuracy
def accuracy(testdata):
    orglabel = []
    prelabel = []
    for x in testdata:
        orglabel.append(x.original_label)
        prelabel.append(x.predict_label)
    orglabel = np.array(orglabel)
    prelabel = np.array(prelabel)
    accuracy = accuracy_score(orglabel,prelabel)
    print("Accuracy:", accuracy)

'''    
if __name__ == "__main__" :
    filename = input("input: ")
    dt_test = read_test(filename)
    testdata = data_preprocessing(dt_test,filename)
    prediction = DecisionTree(testdata)
    accuracy(prediction)
'''

'''
# calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Blues')

# set image label
ax.set_xticks(np.arange(2))
ax.set_yticks(np.arange(2))
ax.set_xticklabels(['False', 'True'])
ax.set_yticklabels(['False', 'True'])
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')

# plot accuracy
total = np.sum(cm)
for i in range(2):
    for j in range(2):
        text = f'{cm[i, j]/total:.2%}'
        ax.text(j, i, text,
                ha="center", va="center",
                color="white" if cm[i, j] > total/2 else "black")

# add color bar
cbar = ax.figure.colorbar(im, ax=ax)

# show image
plt.savefig("DTresult.png",dpi=300,bbox_inches='tight')
plt.show()

'''
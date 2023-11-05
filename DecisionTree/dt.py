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


# read test data    
def read_test(filename):
    df_test = pd.read_csv(filename,encoding='utf-8')
    return df_test

# data preprocessing
def data_preprocessing(inputdata,inputfilename):
    testlist = []
    with open(inputfilename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row["filename"]
            testlist.append(dtdata(filename))
    return testlist

# prediction
def DecisionTree(testdata,testcsv):
    # fill the missing value
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    
    X_test = imp.fit_transform(testcsv.drop(['filename', 'header', 'version'], axis=1))
    # y_test = testcsv['Malware']

    # build decision tree model
    model = joblib.load('../PDFproject/DecisionTree/decision_tree_pdf.pkl')
    
    # predict test data
    y_pred = model.predict(X_test)
    
    #index = 0
    for x,y in zip(testdata,y_pred):
        x.predict_label = y
        #index += 1
    return testdata

'''
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

'''    
if __name__ == "__main__" :
    filename = input("input: ")
    dt_test = read_test(filename)
    testdata = data_preprocessing(dt_test,filename)
    prediction = DecisionTree(testdata,dt_test)
    accuracy(prediction)

    y_test = []
    y_pred = []
    
    for x in prediction:
        y_test.append(x.original_label)
        y_pred.append(x.predict_label)
    
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    # print accuracy
    total = len(y_test)
    correct = 0
    for x,y in zip(y_test,y_pred):
        if x == y:
            correct += 1
    
    print(f"Accuracy: {correct/total}")
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
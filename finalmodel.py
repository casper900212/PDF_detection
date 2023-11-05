from DecisionTree import dt
import PDFmodel
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import csv
import os
import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm
import os
import random
from DecisionTree import dt

class bertdata:
    def __init__(self,filename,content):
        self.filename = filename
        self.content = content
        self.original_label = 2
        self.predict_label = 2

if __name__ == "__main__":
    
    testcsvfile = input("Input csv file: ")
    # Decision Tree
    testcsv = dt.read_test(testcsvfile)
    testdata = dt.data_preprocessing(testcsv,testcsvfile)
    dtprediction = dt.DecisionTree(testdata,testcsv)
    
    # Bert
    jsdir = input("Input js directory: ")
    # jsdir2 = input("Input js directory(malicious): ")
    test_dataset = []
    for file in os.listdir(jsdir):
        with open(jsdir+file, encoding='utf-8') as f:
            filename = os.path.splitext(os.path.basename(f.name))[0]
            content = f.read()
            # print(content[:200])
            test_dataset.append(bertdata(filename,content))
    
    '''        
    for file in os.listdir(jsdir2):
        with open(jsdir2+file, encoding='utf-8') as f:
            filename = os.path.splitext(os.path.basename(f.name))[0]
            content = f.read()
            # print(content[:200])
            test_dataset.append(bertdata(filename,content,1))
    '''        
    
    '''
    zxbenign_test_dataset = []
    for file in os.listdir('./dataset/zx_benign_test'):
        with open('./dataset/zx_benign_test/'+file, encoding='utf-8') as f:
            filename = os.path.splitext(os.path.basename(f.name))[0]
            content = f.read()
            # print(content[:200])
            zxbenign_test_dataset.append(bertdata(filename,content,0))
            
    zxmal_test_dataset = []
    for file in os.listdir('./dataset/zx_malicious_test'):
        with open('./dataset/zx_malicious_test/'+file, encoding="utf-8") as f:
            filename = os.path.splitext(os.path.basename(f.name))[0]
            content = f.read()
            # print(content[:200])
            zxmal_test_dataset.append(bertdata(filename,content,1))
    '''
    
    print("Load model ...")
    loaded_model = LongformerForSequenceClassification.from_pretrained("../PDFproject/4096Model")
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    loaded_model.eval()
    
    bertprediction = test_dataset
    random.shuffle(testdata) 

    correct = 0
    count = 0
    totaltest = len(testdata)

    print("Predicting ...")
    for x in bertprediction:
        encoded = tokenizer.encode_plus(
            x.content,
            add_special_tokens=True,
            max_length=4096,
            padding=True,
            truncation = True,
            return_tensors="pt"
        )

        # Make predictions on the new data
        with torch.no_grad():
            inputs = {key: value.to(device) for key, value in encoded.items()}
            outputs = loaded_model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            x.predict_label = int(predictions)
    
    # filename rename
    for x in bertprediction:
        if ".js" in x.filename:
            x.filename = x.filename.replace(".js","")
    
    '''
    # compare prediction result
    for x in bertprediction:
        for y in dtprediction:
            if x.filename in y.filename:
                y.predict_label = x.predict_label   
             
    # evaluate accuracy
    print("Evaluating ...")
    correct = 0
    total = len(dtprediction)
    for x in dtprediction:
        if x.original_label == x.predict_label:
            correct += 1
    accuracy = correct/total
    print("Accuracy: ",correct," / ",total,"\t",accuracy)
    
    # plot confusion matrix
    print("Plotting result ...")
    probability = [accuracy, 1 - accuracy]
    labels = ['success','fail']

    # plot the result
    orglabel = []
    prelabel = []
    for x in dtprediction:
        orglabel.append(x.original_label)
        prelabel.append(x.predict_label)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    cm = confusion_matrix(orglabel, prelabel)
    im = ax1.imshow(cm, cmap='Blues')
    
    ax1.set_xticks(np.arange(2))
    ax1.set_yticks(np.arange(2))
    ax1.set_xticklabels(['False', 'True'])
    ax1.set_yticklabels(['False', 'True'])
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix')
    
    # plot accuracy
    sum = np.sum(cm)
    for i in range(2):
        for j in range(2):
            text = f'{cm[i, j]/total:.2%}'
            ax1.text(j, i, text,
                ha="center", va="center",
                color="white" if cm[i, j] > sum/2 else "black")

    # add color bar
    cbar = ax1.figure.colorbar(im, ax=ax1)

    ax2.pie(probability, labels=labels, autopct='%1.1f%%', startangle=140)
    ax2.set_title('Accuracy')

    plt.tight_layout()
    plt.savefig('final.png')
    #f1 = f1_score(prelabel, orglabel)
    #print("F1_score: ",f1)
    '''
    
    #write result to the csv file
    print("Writing result to csv file ...")
    output = input("output filename: ")
    header = ['filename','predict_label']
    with open(output,'w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        row = ['filename','predict_label']
        for x in dtprediction:
            row[0] = os.path.basename(x.filename)
            row[1] = x.predict_label
            writer.writerow(row)
        
    
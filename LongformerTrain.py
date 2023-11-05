import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os

# Load a pre-trained BERT model and tokenizer
model_name = "allenai/longformer-base-4096"
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerForSequenceClassification.from_pretrained(model_name, gradient_checkpointing=True, num_labels=2)


# self defined structure for data
class bertdata:
    def __init__(self,filename,content,original_label):
        self.filename = filename
        self.content = content
        self.original_label = int(original_label)
        self.predict_label = 2

# 2975
print("Load data ...")
benign_dataset = []
for file in os.listdir('/home/ubuntu/ndisk/PDFproject/dataset/de_benign_9109'):
    with open('/home/ubuntu/ndisk/PDFproject/dataset/de_benign_9109/'+file, encoding="utf-8") as f:
        filename = os.path.splitext(os.path.basename(f.name))[0]
        content = f.read()
        # print(content)
        benign_dataset.append(bertdata(filename,content,0))

benignsize = len(benign_dataset)

# 6646

mal_dataset = []
for file in os.listdir('/home/ubuntu/ndisk/PDFproject/dataset/de_malicious_30071'):
    with open('/home/ubuntu/ndisk/PDFproject/dataset/de_malicious_30071/'+file) as f:
        filename = os.path.splitext(os.path.basename(f.name))[0]
        content = f.read()
        # print(content[:200])
        mal_dataset.append(bertdata(filename,content,1))
            
malicioussize = len(mal_dataset)

# 234
zxmal_train_dataset = []
for file in os.listdir('/home/ubuntu/ndisk/PDFproject/dataset/zx_malicious_train'):
    with open('/home/ubuntu/ndisk/PDFproject/dataset/zx_malicious_train/'+file, encoding="utf-8") as f:
        filename = os.path.splitext(os.path.basename(f.name))[0]
        content = f.read()
        # print(content[:200])
        zxmal_train_dataset.append(bertdata(filename,content,1))
        
# 116        
zxmal_test_dataset = []
for file in os.listdir('/home/ubuntu/ndisk/PDFproject/dataset/zx_malicious_test'):
    with open('/home/ubuntu/ndisk/PDFproject/dataset/zx_malicious_test/'+file, encoding="utf-8") as f:
        filename = os.path.splitext(os.path.basename(f.name))[0]
        content = f.read()
        # print(content[:200])
        zxmal_test_dataset.append(bertdata(filename,content,1))

    # 281
zxbenign_train_dataset = []
for file in os.listdir('/home/ubuntu/ndisk/PDFproject/dataset/zx_benign_train'):
    with open('/home/ubuntu/ndisk/PDFproject/dataset/zx_benign_train/'+file, encoding='utf-8') as f:
        filename = os.path.splitext(os.path.basename(f.name))[0]
        content = f.read()
        # print(content[:200])
        zxbenign_train_dataset.append(bertdata(filename,content,0))

# 140
zxbenign_test_dataset = []
for file in os.listdir('/home/ubuntu/ndisk/PDFproject/dataset/zx_benign_test'):
    with open('/home/ubuntu/ndisk/PDFproject/dataset/zx_benign_test/'+file, encoding='utf-8') as f:
        filename = os.path.splitext(os.path.basename(f.name))[0]
        content = f.read()
        # print(content[:200])
        zxbenign_test_dataset.append(bertdata(filename,content,0))


traindata = benign_dataset + mal_dataset + zxmal_train_dataset + zxbenign_train_dataset
traindata_content = []
traindata_label = []
# validationdata_content = []
# validationdata_label = []

for x in traindata:
    traindata_content.append(x.content)
    traindata_label.append(x.original_label)
    
print("data split ...")
# Split the data into train and validation sets
train_content, val_content, train_labels, val_labels = train_test_split(
    traindata_content, traindata_label, test_size=0.2, random_state=42
)
    
# Tokenize and prepare the data
print("data preprocessing ...")
input_ids = []
attention_masks = []


class JavaScriptDataset():
    def __init__(self, code_list, label_list, tokenizer, max_length):
        self.code_list = code_list
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.code_list)

    def __getitem__(self, idx):
        code = self.code_list[idx]
        label = self.label_list[idx]

        inputs = self.tokenizer.encode_plus(
            code,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }
'''
encoded = tokenizer.encode_plus(
    traindata_content,
    add_special_tokens=True,
    max_length=512,
    padding='max_length',
    truncation=True,
    # return_attention_mask=True,
    return_tensors="pt",
)
'''
# input_ids.append(encoded["input_ids"])
# attention_masks.append(encoded["attention_mask"])

# Create DataLoader objects for training and validation
train_data = JavaScriptDataset(train_content, train_labels, tokenizer, max_length=4096)
train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True,pin_memory = True)
val_data = JavaScriptDataset(val_content, val_labels, tokenizer, max_length=4096)
val_dataloader = DataLoader(val_data, batch_size=2, shuffle=True,pin_memory = True)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * 3  # 3 epochs
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training loop
num_epochs = 3
model.to(device)

num_training_steps=len(train_dataloader)
print("training ...")
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    print(f"Epoch{epoch+1} ...")
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        
        progress_bar.update(1)

    # Calculate average loss for this epoch
    avg_train_loss = total_loss / len(train_dataloader)
    
    progress_bar.reset()
    # Validation
    model.eval()
    val_predictions = []
    val_true_labels = []

    for batch in val_dataloader:
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

        val_predictions.extend(predictions.tolist())
        val_true_labels.extend(labels.tolist())

    val_accuracy = accuracy_score(val_true_labels, val_predictions)

    print(f"Epoch {epoch + 1}/{num_epochs} - Avg Training Loss: {avg_train_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")

# Save the fine-tuned model
print("Save model ...")
save_directory = "./Longformer4096/"
model.save_pretrained(save_directory)

'''
print("Load model ...")
# Load the saved model for prediction
loaded_model = LongformerForSequenceClassification.from_pretrained('/home/ubuntu/ndisk/unlimiformer/unlimiformer-main/src/4096Model/')
loaded_model.to(device)
loaded_model.eval()
'''

'''
def classify_code(code, tokenizer, model):
    model.eval()
    inputs = tokenizer.encode_plus(
        code,
        None,
        add_special_tokens=True,
        max_length=4096,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = loaded_model(**inputs)

    logits = outputs.logits
    probabilities = torch.relu(logits)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    return predicted_label

f1_scores = []
'''

'''
# Data for prediction (replace with your own data)
testdata = zxbenign_test_dataset + zxmal_test_dataset 

# Tokenize and prepare the new data
new_input_ids = []
new_attention_masks = []

correct = 0
count = 0
totaltest = len(testdata)

print("Predicting ...")
for x in testdata:
    encoded = tokenizer.encode_plus(
        x.content,
        add_special_tokens=True,
        max_length=4096,
        padding='max_length',
        truncation=True,
        return_tensors="pt",
    )

    # Make predictions on the new data
    with torch.no_grad():
        inputs = {key: value.to(device) for key, value in encoded.items()}
        outputs = loaded_model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        x.predict_label = int(predictions)
    
    if(x.predict_label == x.original_label):
        correct += 1
    #print(f"test{count} Predicted label: {predicted_label} \tOriginal label: {testdata_label[count]}\n")
    count += 1
    print(f"total test data: {totaltest}")
    
accuracy = correct / totaltest
print("Accuracy: ",correct," / ",len(testdata),"\t",accuracy)
'''

'''
testdata = zxbenign_test_dataset + zxmal_test_dataset
totaltest = len(testdata)
testdata_content = []
testdata_label = []
for x in testdata:
    testdata_content.append(x.content)
    testdata_label.append(x.original_label)

# evaluate accuracy
predict_labels = []
correct = 0
count = 0       # counter for testdata_label index
for x in testdata_content:
    predicted_label = classify_code(x, tokenizer, model)
    predict_labels.append(predicted_label)
    if(predicted_label == testdata_label[count]):
        correct += 1
    print(f"total test data: {totaltest}")
    print(f"test{count} Predicted label: {predicted_label} \tOriginal label: {testdata_label[count]}\n")
    count += 1

accuracy = correct / len(testdata)
print("Accuracy: ",correct," / ",len(testdata),"\t",accuracy)

probability = [accuracy, 1 - accuracy]
labels = ['success','fail']

# plot the result
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
cm = confusion_matrix(testdata_label, predict_labels, labels = [0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= [0, 1])
disp.plot(ax=ax1)
ax1.set_title('Confusion Matrix')

ax2.pie(probability, labels=labels, autopct='%1.2f%%', startangle=140)
ax2.set_title('Accuracy')

plt.tight_layout()
plt.savefig(f'Fold1.png')
f1 = f1_score(predict_labels, testdata_label)
f1_scores.append(f1)
print("F1_score: ",f1)
'''

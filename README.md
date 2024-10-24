# # 惡意檔案過濾系統建置手冊
## **EXE 系統建置**
* **前置作業**
    * 下載 /exe 整個資料夾到要運行的環境
    (假設檔案路徑為 /exe)
    * 安裝 docker
        > sudo apt-get install docker.io
    * 進到 /exe
        > cd /exe

* **Build image**
    ```
    sudo docker build -t="[your-name/image-name]" .
    
    # 要 commit 到 Docker Hub 的話，會需要針對該 Docker Image 製作一個TAG，之後使用這個 TAG 做push
    ```
    Example:
    ```
    sudo docker build -t="zyxel/exe" .
    ```
* **Create a container**
    ```
    sudo docker run -it --gpus all -v [本機位置]:[container 路徑] --name [你想取的名字] [image] [command]
    ```

    Example:
     ```
     sudo docker run -it --gpus all -v ~NCUML_Prefilter/exe:/exe --name zyxel_exe zyxel/exe:latest /bin/bash 
     ```

* **Run Decision Tree**

    ```
    #進入 docker
     sudo docker start zyxel_exe
     sudo docker attach zyxel_exe
    
    #進入檔案存放位置
     cd exe/decision/
    
    #preprocess 輸入的檔案位置結尾要有 "/"
    
    #run preprocess_train.py (只對training data 做提取)
    #提取malicious 和 benign data features，會在 ./dataset/ 產生 dataset_malware.csv dataset_clean.csv
     python preprocess_train.py [train_mal 位置] [train_bn 位置]
    
    #run preprocess_test.py
    #提取test data features，會在 ./output/ 產生 testdata.csv
     python preprocess_test.py [testdata 位置]
    
    #run merge.py 
    #合併 malware data 和 benign data，會在 ./dataset/ 產生 merge_train.csv
     python merge.py
    
    #run model.py (產生 Decision Tree Model)
     python model.py [model name](不含副檔名)
     
    #run predict.py (預測 testdata 是否為 malware)
     python predict.py [model file](含副檔名)
    
    #預測出來的結果會在 /exe/decision/result 中，檔案名稱為 predict.csv
    ```

* **備注**
    * About Data : 
    ```
        |-- exe
        |   -- data
        |      -- clean          #training benign data
        |      -- malware        #training malicious data
        |      -- testbn         #testing benign data
        |      -- testmw         #testing malicious data
        |   -- decision
        |      -- dataset        #加上 malicious label & 合併完 malicious 和 benign 的 csv 檔案位置
        |      -- output         #分別提取完 features 產生 csv 檔案的位置 (沒有 malicious label)
        |      -- result         #predict 結果存放位置
        |   -- DecisionTree.py   #在 training data 和 testing data 都已知的情況下 驗證預測效果
        |   -- exe.joblib        #Decision Tree model
        |   -- preprocess_all    #在 training data 和 testing data 都已知的情況下 提取 features
        |   ...
    ```
    * About Training : 
        如果要用新的資料集做 training ，並***檢查模型預測準確率***的話，步驟如下：
        ```
        #run preprocess_all.py (提取 training data 和 "分類好的" testing data)
        #會在 ./dataset/ 產生 dataset_malware.csv dataset_clean.csv dataset-testmw.csv dataset-testbn.csv
         python preprocess_all.py [train_mal位置] [train_bn位置] [test_mal位置] [test_bn位置]
         
        #run DecisionTree.py 
         python DecisionTree.py
         
        #會產生 predict.csv 和 confusion.png (Confusion Matrix) 來查看模型訓練效果
        ```
    
-------------------------------------------------------------------------

## **PDF系統建置**
* **前置作業**
    * 下載 /PDFproject 整個資料夾到要運行的環境
    (假設檔案路徑為 /PDFproject)
    * 安裝 docker
        > sudo apt-get install docker.io
    * 進到 /PDFproject
        > cd /PDFproject

* **Build image**
    ```
    sudo docker build -t="[your-name/image-name]" .
    
    # 要 commit 到 Docker Hub 的話，會需要針對該 Docker Image 製作一個TAG，之後使用這個 TAG 做push
    ```
    Example:
    ```
    sudo docker build -t="zyxel/pdf" .
* **Create a container**
     ```
    sudo docker run -it --gpus all -v [本機位置]:[container 路徑] --name [你想取的名字] [image] [command]
    ```

    Example:
     ```
     sudo docker run -it --gpus all -v ~/NCUML_Prefilter/PDFproject:/PDFproject --name zyxel_pdf zyxel/pdf:latest /bin/bash 
     ```

     
* **Run the process**

    ```
    #進入 docker
     sudo docker start zyxel_pdf
     sudo docker attach zyxel_pdf
    
    #進行PDF的features抽取
       cd /PDFproject/DecisionTree/pdfid
       #run extractfeatures.py
       python extractfeatures.py
       #抽feature時會需要輸入需抽取的PDF目錄以及抽完的features要放的位置
       Input PDF directory: /path/to/pdf/
       Output: features.csv
       
    
    #進行PDF的JavaScript抽取 
       cd /PDFproject/extract-js-from-pdf
       #更改extractjs.sh的權限
       sudo chmod 777 extractjs.sh
       # 再抽之前可以先建一個用來存抽完的JS的目錄
       mkdir js
       # run extractjs.sh
       ./extractjs.sh "INPUT_DIRECTORY" "OUTPUT_DIRECTORY"
       # Example
       ./extractjs.sh /path/to/pdf/ /path/to/js/
    
    #進行預測
       # 在預測前若model沒有先pre-train的話需先從release裡把pytorch_model_bin載下來放進Longformer4096的目錄下解壓縮
       cd /PDFproject/
       #run finalmodel.py 
       python finalmodel.py
       # 執行finalmodel時會需要輸入剛剛抽完的feature檔跟JS目錄以及輸出預測結果的檔案
       input csv file: /path/to/csv/features.csv
       input js directory: /path/to/js/ 
       output file: predict.csv
    ```

* **Train the model**

    ```
    # Decision Tree
    cd /PDFproject/DecisionTree
    #run DTtrain.py
    python DTtrain.py
    # train完的model為decision_tree_pdf.pkl
    # 若公司之後要重新訓練資料且資料路徑有更動時，可以直接改DTtrain.py裡讀檔的那行code
    # model 參數
    criterion: 衡量模型效能，default為gini
    max_depth: Tree的最大深度
    min_sample_split: 每個interal node最少要分出的sample數
    min_samples_leaf: 葉節點最少要有幾個samples
    max_features: 做分類時參考的最大feature數量
    
    # Longformer
    cd /PDFproject
    #run LongformerTrain.py
    python LongformerTrain.py
    # train完的model會存在Longformer4096那個目錄下
    # Train之前要先把dataset裡的zip檔解壓縮，但如果公司要自己重分訓練資料集的話在LongformerTrain.py裡我有註解一段code是讓公司可以讀自己的資料集的
    # model 參數
    optimizer: 用於更新model的weight跟bias，常用的為Adam
    learning rate: 每次更新參數時的幅度，太大可能會導致model無法收斂
    epoch: model訓練整個資料集的迭帶次數
    ```

* **備註**
    * 這個版本是之後實務運作上不知道收到的PDF是benign還是malicious情況下的版本
    * 有些PDF抽js的時間過長，在extractjs.sh中有設置3秒的timeout，timeout的設置可視之後的實務情況做調整
    * About file : 
    ```
        |-- PDFproject
        |   -- DecisionTree
        |      -- pdfid
        |         -- extractfeatures.py   # 抽取pdf的features
        |         -- FillValueInCSV.py    # 在抽完features的csv file中加上'Malware' column
        |      -- dt.py                   # 訓練完的Decision Tree model,用來整合至finalmodel
        |      -- DTtrain.py              # 訓練Decision Tree model
        |      -- decision_tree_pdf.pkl   # 訓練完的Decision Tree model,dt.py會用到
        |      -- traindata.csv           # 抽完features的training data,用來訓練Decision Tree model
        |      -- testdata.csv            # 抽完features的testing data,用來測試訓練完的Decision Tree model
        |   -- Longformer4096             # 訓練完的Longformer model
        |   -- dataset                    
        |      -- AllPDF                  # 放原始的PDF file
        |      -- de_benign_9109.zip      # 抽完JS的benign dataset
        |      -- de_malicious_30071.zip  # 抽完JS的malicious dataset
        |      -- zx_benign_test.zip      # 抽完JS的zyxel benign test dataset
        |      -- zx_benign_train.zip     # 抽完JS的zyxel benign train dataset
        |      -- zx_malicious_test.zip   # 抽完JS的zyxel malicious test dataset
        |      -- zx_malicious_train.zip  # 抽完JS的zyxel malicious train dataset
        |   -- extract-js-from-pdf
        |      -- extractjs.sh            # 抽取pdf的JS 
        |   -- LongformerTrain.py         # 訓練Longformer model
        |   -- finalmodel.py              # 整合完Decision Tree與Longformer的最終model,用來預測新資料
    ```
    * datasets:

    | PDF | Our PDF | Zyxel PDF |
    | -------- | -------- | -------- |
    | Benign         |  2975        |  2351(train: 1561, test: 790)        |
    | Malicious    | 30097    | 870(train: 573, test: 297)     |

    

    | JavaScript | Our JS | Zyxel JS |
    | ---------- | -------- | -------- |
    | Benign           | 2975         |  421(train: 281, test: 140)        |
    | Malicious       | 10962    | 350(train: 234, test: 116)     |

    

    | JavaScript | Our JS | Zyxel JS |
    | ---------- | -------- | -------- |
    | Benign           | 2975         |  421(train: 281, test: 140)        |
    | Malicious       | 10962    | 350(train: 234, test: 116)     |

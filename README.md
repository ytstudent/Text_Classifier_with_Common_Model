# Text_Classifier_with_LSTM  

A text-classifier with LSTM model,just for rookie like me,It's running with cpu  
I.About data:  

    a.There are only 20 sentences in the data files because the original data is the company's business secret.The data just for you to         understand codes in "type_transforms.py".  
    
    b.Those sentences has been tokenized before I transform it in "data_all.txt".You can use jieba to tokenize Chinese string.  
II.About accuracy: It's very dependent on the original data  
III.About disadvantage : The disadvantage is that I didn't use "early stop" when trainng the model. In addition,it's obvious that I didn't      adjust the parameters. 

IV.Environment：
    python==3.6.7  
    numpy==1.17.2  
    pandas==0.25.1  
    scikit-learn==0.21.3  
    torch==1.3.0+cpu  
    
Last but not least ：It's very easy,just for rookie like me!

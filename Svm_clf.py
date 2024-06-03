from sklearn import svm
from dataset_pre.dataset_load import load_cvs_dataset
from feature_eng.ngram_tf_idf import ngram_tf_idf
from classifier.Classifier import train_model
import os
import pandas as pd

def load_data(directory):
    data = []
    labels = []
    categories = os.listdir(directory)
    for category in categories:
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data.append(file.read())
                    labels.append(category)
    return pd.DataFrame({'text': data, 'label': labels})

def main():
   
    # load the dataset
    trainDF = load_data("../archive")
   
    # load the dataset
 
    # trainDF = load_cvs_dataset("../corpus.csv")
    # load the dataset
    
    # Text Preprocessing
    txt_label = trainDF['label']
    txt_text = trainDF['text']
    
    # Text Preprocessing
   
    # Text feature engineering 
    model_input = ngram_tf_idf(txt_text, txt_label)
    # Text feature engineering 
    
    #  Build Text Classification Model and Evaluating the Model
    naive = svm.SVC()
    accuracy = train_model(naive, model_input[0], model_input[1], model_input[2], model_input[3])
    print ("Svm_clf, ngram_tf_idf accuracy is : ", accuracy * 100)
   
    
if __name__ == '__main__':
    main()

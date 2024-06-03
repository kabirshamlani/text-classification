from sklearn import linear_model
from dataset_pre.dataset_load import load_cvs_dataset
from feature_eng.word_tf_idf import word_tf_idf
from feature_eng.ngram_tf_idf import ngram_tf_idf
from feature_eng.count_vectorizer import count_vectorizer
from feature_eng.char_tf_idf import char_tf_idf
from classifier.Classifier import train_model
import pandas as pd
import os

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
    # trainDF = load_cvs_dataset("../corpus.csv")
    # load the dataset
    
    # Text Preprocessing
    txt_label = trainDF['label']
    txt_text = trainDF['text']
    #clear_txt = prepare_dataset().clean_cvs_txt(txt_text)
    # Text Preprocessing
    
    # Text feature engineering 
    model_input = char_tf_idf(txt_text, txt_label)
    # Text feature engineering 
    
    #  Build Text Classification Model and Evaluating the Model
    naive = linear_model.LogisticRegression(max_iter=1000)
    accuracy = train_model(naive, model_input[0], model_input[1], model_input[2], model_input[3])
    print ("Linear_Clf, char_tf_idf accuracy is : ", accuracy * 100)
    
    # Text feature engineering 
    model_input = count_vectorizer(txt_text, txt_label)
    # Text feature engineering 
    
    #  Build Text Classification Model and Evaluating the Model
    naive = linear_model.LogisticRegression(max_iter=1000)
    accuracy = train_model(naive, model_input[0], model_input[1], model_input[2], model_input[3])
    print ("Linear_Clf, count_vectorizer accuracy is : ", accuracy * 100)
    
    # Text feature engineering 
    model_input = ngram_tf_idf(txt_text, txt_label)
    # Text feature engineering 
    
    #  Build Text Classification Model and Evaluating the Model
    naive = linear_model.LogisticRegression(max_iter=1000)
    accuracy = train_model(naive, model_input[0], model_input[1], model_input[2], model_input[3])
    print ("Linear_Clf, ngram_tf_idf accuracy is : ", accuracy * 100)
    
    # Text feature engineering 
    model_input = word_tf_idf(txt_text, txt_label)
    # Text feature engineering 
    
    #  Build Text Classification Model and Evaluating the Model
    naive = linear_model.LogisticRegression(max_iter=1000)
    accuracy = train_model(naive, model_input[0], model_input[1], model_input[2], model_input[3])
    print ("Linear_Clf, word_tf_idf accuracy is : ", accuracy * 100)
    
    
if __name__ == '__main__':
    main()
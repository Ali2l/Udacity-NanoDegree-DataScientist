"""
1. Import libraries and load data from database.
"""
import re
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download(['punkt', 'wordnet','omw-1.4'])




def load_data(database_filepath):
    """
    Loading the data from SQLite.
    """

    # load data from database 
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y


"""
2. Write a tokenization function to process your text data
"""
def tokenize(text):
    """
    Text Tokenization Function.
    """
    # text normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    stop_words = stopwords.words("english")

    # tokenize text
    words = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #stemming
    stemmed = [PorterStemmer().stem(w) for w in words]

    #lemmatizing
    lemmed_words = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
        
    return lemmed_words

"""
3. Build a machine learning pipeline¶
"""
def build_model():
    """
    classifier bulider and using GridSearchCV.
    """

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
        
    parameters = {
        'clf__estimator__n_estimators' : [50, 100]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    # return pipeline
    return cv

"""
5. Test your model
"""
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model 
    """
    y_prediction = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_prediction[:, index]))


def save_model(model, model_filepath):
    """ 
    Exports model as a pickle file.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Build, train, evaluat and save the model
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
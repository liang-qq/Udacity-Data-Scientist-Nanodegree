import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import pandas as pd
import numpy as np
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    """
    load data from the sql database into a dataframe

    Args:
    database_filepath: string. File path for the sql database.

    Returns:
    X: datafame. Dataframe for features which will be used for the model.
    Y: datafame. Dataframe for labels which will be used for the model.
    category_names: string list. List that contains column names of the labels dataframe (Y).
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponse", engine)
    df = df.dropna()
    # generate X, Y
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    NLP Process: tokenize and lemmatize texts.

    Args:
    text: string. Raw text for NLP processing.

    Returns:
    tokens: strings. Processed text after tokenize and lemmatize.
    """
    #Write a tokenization function to process your text data
    # Todo: normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Todo: tokenize text
    tokens = word_tokenize(text)
    
    # Todo: lemmatize and remove stop words
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]

    return tokens


def build_model():
    """
    build a machine learning model.

    Args:
    None.

    Returns:
    cv: grid search cv object that is used for tuning machine learning model parameters.
    """
    #use the best parameters to retest the pipeline
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer(use_idf=False)),
                ('clf', MultiOutputClassifier(DecisionTreeClassifier(criterion='gini', max_depth=3)))
    ])
    
    #create grid search obeject to tuning parameters
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_depth': [None, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

#define a function to evaluate accuracy using sklearn.classification_report package
def accuracy_score(Y_pred,Y_test):
    """
    evalution metrics for the machine learning model performance

    Args:
    Y_pred: strings. Predicted labels of the test dataset from the machine learning model.
    Y_test: strings. Real labels for the test dataset.

    Returns:
    None
    """
    for i, col in enumerate(Y_test):
        print(col, '\t', classification_report(Y_test[col], Y_pred[:, i]))

def evaluate_model(model, X_test, Y_test, category_names):
    """
    evalute the machine learning model performance

    Args:
    model: object. A developed machine learning model.
    X_test: strings. Features for the test dataset.
    Y_test: strings. Real labels for the test dataset.
    category_names: string list. List that contains column names of the labels dataframe (Y).

    Returns:
    None
    """
    #make a prediction
    Y_pred = model.predict(X_test)
    #evaluate model accuracy
    accuracy = accuracy_score(Y_pred, Y_test)



def save_model(model, model_filepath):
    """Pickle fitted model
    
    Args:
    model: object. A developed machine learning model.
    model_filepath: string. Filepath for where the developed machine learning model is saved
    
    Returns:
    None
    """
    # Pickle best model
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
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
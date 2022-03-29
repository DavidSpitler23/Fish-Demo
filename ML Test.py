# Imports
import os
import pandas as pd
import numpy as np
from sklearn import model_selection, linear_model, neighbors, ensemble

# Constants
PATH_TO_TRAINING_DIRECTORY = './' # Path to training data
DATA_NAME = 'Fish.csv' # Name of training data to access
NUM_SPLITS = 5 # Number of splits/folds for model training

if __name__ == '__main__':

    # Load training data into numpy array
    df = pd.read_csv(os.path.join(PATH_TO_TRAINING_DIRECTORY, DATA_NAME), index_col=0)
    # index_col = row number of data, start at 0 when it's actually data
    print(df.head())
    matrix = df.to_numpy()
    
    # Separate array into labels and data
    y = matrix[:,0]
    X = matrix[:,1:]
    # print(y, X)
     
    # Train and score models using cross validation
    k_fold = model_selection.KFold(n_splits=NUM_SPLITS, shuffle=True)
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    # Shuffles data, splits into 5 sections. In the loop, each section will be used as train/test data at some point.
    # This ensures that any important data isn't missed

    logistic_scores = 0
    
    for train_index, test_index in k_fold.split(X):
        # Splits into training and testing data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Training logistic regression model
        log_model = linear_model.LogisticRegression()
        log_model.fit(X_train, y_train)
        logistic_scores += log_model.score(X_test, y_test)

    # Calculates average score of models
    avg_log_score = logistic_scores/NUM_SPLITS

    # Display results
    print('Average success rate of logistic regression model: {}%'.format(avg_log_score*100))
    
    # print('Fish Identification:')
    
    # while i==1:
    #     Weight_test=input('Weight: ')
    #     Length1_test=input('Length: ')
    #     Length2_test=input('Length2: ')
        
    
    
    

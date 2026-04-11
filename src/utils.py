import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        # Ensure the destination folder exists before writing the serialized object.
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        # Persist Python objects such as preprocessors or trained models to disk.
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            # Run hyperparameter search for each model using the provided grid.
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            # Refit the model with the best parameter combination found by GridSearchCV.
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # Evaluate the tuned model on both training and test sets.
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test R2 score so callers can compare all models easily.
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        # Load serialized objects back into memory for prediction or reuse.
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

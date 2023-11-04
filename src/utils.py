import os
import sys
import dill as pickle
import numpy as np
import pandas as pd
from src.exceptions import Customed_exception
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def SaveObject(file_path, obj):
    try:
        Dpath = os.path.dirname(file_path)
        os.makedirs(Dpath, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    except Exception as e:
        raise Customed_exception(e, sys)

def model_evaluation(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=4)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise Customed_exception(e, sys)

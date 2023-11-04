import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import SaveObject
from src.exceptions import Customed_exception
from src.logs import logging
from src.utils import model_evaluation

@dataclass
class ModelTrainerConfig:
    train_model_path =  os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.ModelTrainerConfig = ModelTrainerConfig()
    def initiate_trainer(self, train_array, test_array):
        try:
            logging.info("train, test, split input data")   
            X_train, y_train, X_test, y_test = (train_array[:,:-1], 
                                                train_array[:,-1],
                                                test_array[:,:-1],
                                                test_array[:,-1]
                                                )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),}
            
            model_report:dict= model_evaluation(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test, models=models,param=False)
            best_score = max(sorted(model_report.values())) 
            best_model = list(model_report.keys())[list(model_report.values()).index(best_score)]
            best_model = models[best_model]

            if best_score<=0.8:
                raise Customed_exception("best model not found")
            logging.info("best model found using testing data")

            SaveObject(
                file_path = self.ModelTrainerConfig.train_model_path,
                obj = best_model
            )
            predicted = best_model.predict(X_test)
            r2score = r2_score(y_test, predicted)
            return r2score
        except Exception as e:
            raise Customed_exception(e, sys)
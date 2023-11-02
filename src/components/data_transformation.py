import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exceptions import Customed_exception
from src.logs import logging
from sklearn.compose import ColumnTransformer #use for pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import SaveObject

@dataclass
class ConfigDataTransformation:
    preprocessor_path = os.path.join('artifacts',"DataTransformer.pkl")

class datatransformation:
    def __init__(self):
        self.data_transformation_config = ConfigDataTransformation()
    
    def Get_dataTransformer_obj(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        except Exception as e:
            raise Customed_exception(e,sys)
    
    def initiate_DT(self, train_path, test_path): #initiating data transformation

        try:
            logging.info("reading training and testing data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("completed reading training and testing data")

            logging.info("Getting Data ready")
            preprocessor_obj = self.Get_dataTransformer_obj()

            traget_col_name = "math_score"
            numerical_col_name = ["writing_score", "reading_score"]

            input_features_train = train_df.drop(columns =[traget_col_name], axis=1)
            target_features_train = train_df[traget_col_name]

            input_features_test = test_df.drop(columns =[traget_col_name], axis=1)
            target_features_test = test_df[traget_col_name]

            logging.info("applying preprocessing techniques on data")

            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train)
            input_features_test_arr = preprocessor_obj.transform(input_features_test)

            train_arr = np.c_[input_features_train_arr, np.array(target_features_train)]
            test_arr = np.c_[input_features_test_arr, np.array(target_features_test)]

            logging.info(f"preprocessing completed")
            
            SaveObject(
                file_path = self.data_transformation_config.preprocessor_path,
                obj = preprocessor_obj
                )
            logging.info("Data transformation completed")
            return(
                test_arr,
                train_arr,
                self.data_transformation_config.preprocessor_path)
            
        except Exception as e:
            raise Customed_exception(e, sys)

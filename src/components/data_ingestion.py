import os
import sys
from src.exceptions import Customed_exception
from src.logs import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import datatransformation
from src.components.trainer import ModelTrainer
from src.components.trainer import ModelTrainerConfig
@dataclass
class ConfigDataIngestion:
    raw_data_path: str= os.path.join('artifacts',"data.csv")
    train_data_path: str= os.path.join('artifacts',"train.csv")
    test_data_path: str= os.path.join('artifacts',"test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = ConfigDataIngestion()
    
    def initiate_DI(self):
        logging.info("initiated data ingestion")
        try:
            df = pd.read_csv("NoteBooks\data\stud.csv")
            logging.info("Dataset Read(imported) as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("inititating train test split")
            train_set, test_set= train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train test split completed")
            logging.info("Data Ingestion Completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise Customed_exception(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_DI()

    datatransform = datatransformation()
    train_array, test_array,_=datatransform.initiate_DT(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_trainer(train_array, test_array))
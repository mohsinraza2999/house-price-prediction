import pandas as pd
from pathlib import Path
from src.utils.logs import logger
from src.utils import config_utils

class Readcsv:
    def __init__(self,path:Path) ->None:
        self.path=path
    def read(self,name:str)-> pd.DataFrame:
        # reading and returning the csv file
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")
        logger.info("Data Loading.")
        return pd.read_csv(self.path/name)
        

class CleanData(Readcsv):
    def __init__(self, path: Path) -> None:
        super().__init__(path)

    def clean(self,name:str,test=False) ->None:
        # calling the read method of the super class to read csv file
        data=self.read(name)
        logger.info("Data Loading Complete.")
        # selecting the only column of the series
        data=data.iloc[:, 0]
        # calling the seprate_columns method to seprate all the columns values and making a DataFrame
        column_names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        formatted_data=pd.DataFrame(data=self.seprate_columns(data),columns=column_names, dtype=float)
        # droping missing values in rows
        clean_data=self.drop_missing_values(formatted_data)
        # converting all the columns datatype to float
        structured_data=self.dtype_to_float(clean_data)
        logger.info("Data Structured complete.")
        # saving the the processed data in csv file
        if test==True:
            self.save_data(structured_data)
            logger.info("Data Saved.")

    def dtype_to_float(self,data:pd.DataFrame) ->pd.DataFrame:

        for col in data.columns:
            data[col] = pd.to_numeric(data[col])
        return data


    def drop_missing_values(self,data:pd.DataFrame) -> pd.DataFrame:
        return data.dropna(axis=0)

    def seprate_columns(self,data:pd.DataFrame) ->list[str]:
        record_points=[]
        for record in data:
            
            record_points.append(record.replace("  "," ").replace('  ',' ').split(' ')[1:])

        return record_points
    
    def save_data(self,data:pd.DataFrame):
        path_cfg=config_utils.paths_config()['process_data']
        data.to_csv(Path(path_cfg['path'])/path_cfg['name'],index=False)

#if __name__ == "__main__":
def data_pipeline()-> None:
    # creating CleanData class object
    path_cfg=config_utils.paths_config()['raw_data']

    cleaning_object=CleanData(Path(path_cfg['path']))

    # calling the clean method to clean data using the created object
    cleaning_object.clean(name=path_cfg['name'],test=True)
from dataclasses import dataclass
import os
import pandas as pd
from typing import Union
from pydantic import BaseModel

class DataLoader(BaseModel):
    file_name: str = 'train.csv.zip'
    data_folder_name: str = 'data'
    project_root: str = os.path.abspath('./')
    # file_path: str =  os.path.join( project_root, data_folder_name, file_name)
    compression_format: str = 'zip'
    target_column_name: str = 'label'

    def _create_file_path(self):
        return os.path.join(self.project_root, self.data_folder_name, self.file_name)

    def read_data(self) -> pd.DataFrame:
        return pd.read_csv(self._create_file_path(), compression=self.compression_format)
    
    def extract_input_and_target(self, input_df:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        target = input_df[self.target_column_name]
        input_data = input_df.drop(labels=[self.target_column_name], axis=1).reset_index(drop=True)

        return input_data, target
    
if __name__ == '__main__':
    print(os.path.abspath('./'))

import os, sys
import requests
import zipfile
import kaggle
from pathlib import Path

from plant_disease_clf.logger import logging
from plant_disease_clf.exception import CustomException

from plant_disease_clf.entity import DataIngestionConfig
from plant_disease_clf.config.configuration import ConfigManager


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def get_data(self):
        try:

            kaggle_url = self.config.kaggle_URL
            output_dir = self.config.root_dir
            zip_file_path = self.config.kaggle_file

            kaggle_url = kaggle_url.split("/")

            print(kaggle_url[-2])
            print(kaggle_url[-3])
            kaggle_url = kaggle_url[-3] + "/" + kaggle_url[-2]

            # Create the output directory if it doesn't exist
            # os.makedirs(output_dir, exist_ok=True)

            # Download the dataset
            # response = requests.get(kaggle_url)

            # # Save the downloaded dataset to a file
            # dataset_file_path = os.path.join(output_dir, 'dataset.zip')
            # with open(dataset_file_path, 'wb') as file:
            #     file.write(response.content)

            # os.system('kaggle datasets download -d {kaggle_url[-3]}/{kaggle_url[-2]} -p {output_dir}')
            kaggle.api.dataset_download_files(kaggle_url, path=output_dir, unzip=False)

            # Unzip the dataset
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)

            # # Remove the downloaded zip file
            # os.remove(dataset_file_path)

        except Exception as e:
            raise CustomException(e, sys)

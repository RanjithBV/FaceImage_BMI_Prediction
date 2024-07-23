import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from PIL import Image

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_features):
        '''
        This function is responsible for data transformation for numerical features.
        '''
        try:
            # Create a pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_features}")

            # Define the ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def load_image(self, filename):
        image_path = os.path.join(r'C:\bmi\notebook\data\images', filename)
        try:
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
            image = transform(image)
            return image
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, numerical_features):
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Initialize preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(numerical_features)

            target_column_name = "bmi"
            filename_column = "filename"

            # Split into input features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on numerical features.")

            # Transform numerical features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df[numerical_features])
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df[numerical_features])

            logging.info("Loading and transforming images.")

            # Transform images
            train_images = np.array([self.load_image(filename) for filename in train_df[filename_column]])
            test_images = np.array([self.load_image(filename) for filename in test_df[filename_column]])

            # Flatten images
            train_images_flattened = train_images.reshape(len(train_images), -1)
            test_images_flattened = test_images.reshape(len(test_images), -1)

            # Combine numerical and image features
            train_arr = np.hstack((input_feature_train_arr, train_images_flattened))
            test_arr = np.hstack((input_feature_test_arr, test_images_flattened))

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

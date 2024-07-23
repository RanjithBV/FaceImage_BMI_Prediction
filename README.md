# BMI PREDICTION USING FACIAL IMAGES

This project aims to predict Body Mass Index (BMI) from facial images using machine learning techniques. BMI is a measure of body fat based on height and weight that applies to adult men and women. Instead of using the traditional approach of calculating BMI using height and weight measurements, this project leverages facial images to predict BMI, providing a non-invasive and quick method for BMI estimation.

## Dataset Description
The dataset used in this project consists of facial images and their corresponding BMI values. The dataset is publicly available on Kaggle and can be accessed here(https://www.kaggle.com/datasets/nitishkundu/public-bmi-dataset)

Dataset Details
Facial Images: The dataset contains images of individuals' faces. Each image is associated with a unique filename.

BMI Values: Each image has a corresponding BMI value. BMI is calculated as weight in kilograms divided by the square of height in meters (kg/m²).

Columns:

filename: The name of the image file.

BMI: The Body Mass Index of the individual in the image.

Usage:
The facial images are loaded and preprocessed using OpenCV and Keras. The images are resized and normalized before being fed into the CNN model for training and prediction. The BMI values are used as the target variable for model training.

## Webpage link
AWS: http://ec2-3-25-80-164.ap-southeast-2.compute.amazonaws.com

## Technical Aspects
Python 3.9 and more
Important Libraries: sklearn, pandas, numpy, matplotlib, TTensorflow & seaborn
Front-end: HTML, CSS
Back-end: Flask framework
IDE:  Pycharm
Deployment: AWS

## How to run this app
Code is written in Python 3.9 and more. If you don't have python installed on your system, click here https://www.python.org/downloads/ to install.

Create virtual environment - conda create -n venv python=3.9
Activate the environment - conda activate venv
Install the packages - pip install -r requirements.txt
Run the app - python run app.py

1. Data Loading
a. Load Data from CSV and Images
The first step is to load the data from the CSV file and corresponding images from the specified folder.

python
Copy code
def load_data(image_folder, csv_file):
    data = []
    labels = []
    df = pd.read_csv(csv_file)

    print(f"Loaded CSV file: {df.shape[0]} records")

    for index, row in df.iterrows():
        image_path = os.path.join(image_folder, row['filename'])


## Workflow
This workflow describes the process of predicting BMI from facial images, from data loading to model training and saving predictions in a Cassandra database.

### Data Loading

Read the CSV file containing the filenames of the images and their corresponding BMI values.
For each entry in the CSV, load the corresponding image from the specified folder.
Preprocess the images by resizing and converting them to arrays.
Normalize the image data and convert BMI values to a numerical array.
### Data Preprocessing
Split the dataset into training and testing sets to evaluate the model's performance.
### Model Development
Create a Convolutional Neural Network (CNN) model with multiple convolutional layers, pooling layers, and dense layers.
Configure the model with an appropriate optimizer, loss function, and evaluation metrics.
### Model Training
Train the model using the training dataset with a specified number of epochs and batch size.
Validate the model using the testing dataset to monitor performance and avoid overfitting.
### Model Evaluation
Assess the model's performance using Mean Absolute Error (MAE) and other relevant metrics.
### Save the Model
Save the trained model to a file for future use or deployment.
### Store Predictions in Cassandra Database
Connect to Cassandra Database:
Establish a connection to the Cassandra database using secure connection details.
Create Database Table:
Create a table to store the image paths, actual BMI values, and predicted BMI values.
Insert Data into Table:
Insert the predictions along with the image paths and actual BMI values into the Cassandra database.
Query Data:
Verify the data insertion by querying the table and printing the results.

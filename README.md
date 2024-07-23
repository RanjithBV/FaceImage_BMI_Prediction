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

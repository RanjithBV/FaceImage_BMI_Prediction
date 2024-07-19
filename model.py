import os
import cv2
import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import MeanAbsoluteError
from cassandra.cluster import Cluster
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from cassandra.auth import PlainTextAuthProvider
from uuid import uuid4


def load_data(image_folder, csv_file):
    data = []
    labels = []
    df = pd.read_csv('bmi1.csv')

    print(f"Loaded CSV file: {df.shape[0]} records")

    for index, row in df.iterrows():
        image_path = os.path.join(image_folder, row['filename'])
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            image = img_to_array(image)
            data.append(image)
            labels.append(row['BMI'])
        else:
            print(f"Image not found: {image_path}")

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels, dtype="float")

    print(f"Loaded {len(data)} images")
    return data, labels


# Specify your paths
image_folder = 'D:\\ineuron\\BMI_Data\\images'
csv_file = 'D:\\ineuron\\bmi1.csv'
# Load data
data, labels = load_data(image_folder, csv_file)

# Check if data and labels are loaded correctly
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Proceed with data splitting if data is loaded correctly
if data.shape[0] > 0 and labels.shape[0] > 0:
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
else:
    raise ValueError("No data loaded. Check image paths and CSV file.")

# Define the model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    opt = Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=[MeanAbsoluteError()])
    return model


model = create_model()
model.summary()

# Train the model
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=35, batch_size=32)

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))

# Datastax Astra connection details
secure_connect_bundle = 'D:\ineuron\secure-connect-bmi.zip'
client_id = 'HTeWpWGSaCnwZBupBMbylifD'
client_secret = 'Ll2.ZW+MtdgUGocHMUC-eO3hi1WfPxLHrZhms+4ww8NOoCFSAz.W1_o6TBeBQMCe+0WRIqMFWP90gg7iAb_kZYrNZC765648NW-eJAj-5YMNFjmoKq5DY_,yby_HoMvX'


cloud_config = {'secure_connect_bundle': secure_connect_bundle}
auth_provider = PlainTextAuthProvider(client_id, client_secret)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()
session.set_keyspace('rp2')

# Create table if not exists
session.execute("""
    CREATE TABLE IF NOT EXISTS bmi_data (
        id UUID PRIMARY KEY,
        image_path TEXT,
        bmi FLOAT,
        predicted_bmi FLOAT
    )
""")

df = pd.read_csv('bmi1.csv')
predictions = model.predict(testX)

# Insert data into the table
for i in range(len(testX)):
    session.execute("""
        INSERT INTO bmi_data (id, image_path, bmi, predicted_bmi)
        VALUES (%s, %s, %s, %s)
    """, (uuid4(), f"{image_folder}/{df.iloc[i]['filename']}", float(testY[i]), float(predictions[i])))

# Query the data
rows = session.execute("SELECT * FROM bmi_data")
for row in rows:
    print(row)
print("Data inserted into Cassandra successfully.")
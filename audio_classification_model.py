#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# Function to read sounds and convert to spectrograms
def read_data(folder_path):
    labels = []
    spectrograms = []

    for label in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, label)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                if file_path.endswith('.wav'):
                   # print(file_path)
                    y, sr = librosa.load(file_path)
                    S = librosa.stft(y)
                    S_mag = np.abs(S)
                    S_dB = librosa.amplitude_to_db(S_mag, ref=np.max)
                    spectrograms.append(S_dB)
                    labels.append(label)
    
    return spectrograms, labels
# Function to pad or trim a 2D array to a desired shape
def pad2d(a, desired_size):
    rows, cols = a.shape
    padded_a = np.zeros((desired_size, desired_size))
    rows_to_copy = min(rows, desired_size)
    cols_to_copy = min(cols, desired_size)
    padded_a[:rows_to_copy, :cols_to_copy] = a[:rows_to_copy, :cols_to_copy]
    return padded_a

# Create CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

#Function evaluating the model
#def evaluate_model(model, test_data):
    #result = model.evaluate(test_data)
    #print(f'Test Loss: {result[0]}, Test Accuracy: {result[1]}')

# Path to dataset
folder_path = 'C:/Users/Alexandra/Desktop/WORKINGedge-collected-gunshot-audio'

# Read spectrograms and labels
spectrograms, labels = read_data(folder_path)

print(labels)

# Preprocess data
desired_spectrogram_size = 128
spectrograms = np.array([pad2d(s, desired_spectrogram_size) for s in spectrograms])
spectrograms = np.expand_dims(spectrograms, axis=-1)  # Add channel dimension
print(spectrograms.shape)
label_dict = {label: i for i, label in enumerate(set(labels))}
y = np.array([label_dict[label] for label in labels])
y = to_categorical(y)  # One-hot encoding

# Split data
X_train, X_test, y_train, y_test = train_test_split(spectrograms, y, test_size=0.15, random_state=42)

# Define input shape and number of classes
input_shape = X_train[0].shape
num_classes = y.shape[1]

# Create and compile the model
model = create_cnn_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#early stop

earlystopping = EarlyStopping(monitor = 'val_accuracy',
                              mode = 'max' ,
                              patience = 10,
                              verbose = 1)
callback_list = [earlystopping]

# Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks = callback_list, verbose = 1)
model.summary()

# Evaluating the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
# Save the model


# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:09:27 2024

@author: pchri
"""

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk(r'C:\Users\pchri\Downloads'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPooling2D , Flatten , Dropout , BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.callbacks import ReduceLROnPlateau

def get_images(directory):
    images = []
    
    for filename in os.listdir(directory):
        try:
            img = Image.open(os.path.join(directory, filename))
            img = img.resize((128, 128))
            img = img.convert('RGB')
            img = np.array(img) / 255.0
            images.append(img)
        except OSError as e:
            print(f"Error loading {os.path.join(directory, filename)}: {e}")
            continue
    return images

lsd = get_images(r'C:\Users\pchri\Downloads\archive (5)\data\processed_lsd_jpgs')
osf = get_images(r'C:\Users\pchri\Downloads\archive (5)\data\processed_osf_jpgs')
spider = get_images(r'C:\Users\pchri\Downloads\archive (5)\data\processed_spider_jpgs')
tseg = get_images(r'C:\Users\pchri\Downloads\archive (5)\data\processed_tseg_jpgs')

len(lsd)
len(osf)
len(spider)
len(tseg)

fig, ax = plt.subplots(1, 8, figsize=(20, 10))

ax[0].imshow(lsd[0])
ax[1].imshow(lsd[1])
ax[2].imshow(osf[0])
ax[3].imshow(osf[1])
ax[4].imshow(spider[0])
ax[5].imshow(spider[1])
ax[6].imshow(tseg[0])
ax[7].imshow(tseg[1])
ax[0].set_title('lsd')
ax[1].set_title('lsd')
ax[2].set_title('osf')
ax[3].set_title('osf')
ax[4].set_title('spider')
ax[5].set_title('spider')
ax[6].set_title('tseg')
ax[7].set_title('tseg')

for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()

train1_images = np.concatenate((lsd, osf))
train1_labels = np.concatenate((np.ones(len(lsd)), np.zeros(len(osf))))

train1_ds = tf.data.Dataset.from_tensor_slices((train1_images, train1_labels))
batch_size = 32

train1 = train1_ds.shuffle(buffer_size=len(train1_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train1, validation_data = train1, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: lsd v osf')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: lsd v osf')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train1,verbose=2)

pred1 = model.predict(train1)
pred1_cnn = (pred1 > 0.5).astype("int32")

cm1 = confusion_matrix(train1_labels, pred1_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm1, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: lsd v osf')
plt.show()

accuracy1 = accuracy_score(train1_labels, pred1_cnn)
model_report1 = classification_report(train1_labels, pred1_cnn)
print(f'Model accuracy: {round(accuracy1,4)}')
print('Classification Report: lsd v osf')
print(f'{model_report1}')

train2_images = np.concatenate((lsd, spider))
train2_labels = np.concatenate((np.ones(len(lsd)), np.zeros(len(spider))))

train2_ds = tf.data.Dataset.from_tensor_slices((train2_images, train2_labels))
batch_size = 32

train2 = train2_ds.shuffle(buffer_size=len(train2_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train2, validation_data = train2, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: lsd v spider')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: lsd v spider')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train2,verbose=2)

pred2 = model.predict(train2)
pred2_cnn = (pred2 > 0.5).astype("int32")

cm2 = confusion_matrix(train2_labels, pred2_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm2, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: lsd v spider')
plt.show()

accuracy2 = accuracy_score(train2_labels, pred2_cnn)
model_report2 = classification_report(train2_labels, pred2_cnn)
print(f'Model accuracy: {round(accuracy2,4)}')
print('Classification Report: lsd v spider')
print(f'{model_report2}')

train3_images = np.concatenate((lsd, tseg))
train3_labels = np.concatenate((np.ones(len(lsd)), np.zeros(len(tseg))))

train3_ds = tf.data.Dataset.from_tensor_slices((train3_images, train3_labels))
batch_size = 32

train3 = train3_ds.shuffle(buffer_size=len(train3_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train3, validation_data = train3, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: lsd v tseg')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: lsd v tseg')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train3,verbose=2)

pred3 = model.predict(train3)
pred3_cnn = (pred3 > 0.5).astype("int32")

cm3 = confusion_matrix(train3_labels, pred3_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm3, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: lsd v tseg')
plt.show()

accuracy3 = accuracy_score(train3_labels, pred3_cnn)
model_report3 = classification_report(train3_labels, pred3_cnn)
print(f'Model accuracy: {round(accuracy3,4)}')
print('Classification Report: lsd v tseg')
print(f'{model_report3}')

train4_images = np.concatenate((spider, osf))
train4_labels = np.concatenate((np.ones(len(spider)), np.zeros(len(osf))))

train4_ds = tf.data.Dataset.from_tensor_slices((train4_images, train4_labels))
batch_size = 32

train4 = train4_ds.shuffle(buffer_size=len(train4_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train4, validation_data = train4, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: spider v osf')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: spider v osf')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train4,verbose=2)

pred4 = model.predict(train4)
pred4_cnn = (pred4 > 0.5).astype("int32")

cm4 = confusion_matrix(train4_labels, pred4_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm4, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: spider v osf')
plt.show()

accuracy4 = accuracy_score(train4_labels, pred4_cnn)
model_report4 = classification_report(train4_labels, pred4_cnn)
print(f'Model accuracy: {round(accuracy4,4)}')
print('Classification Report: spider v osf')
print(f'{model_report4}')

train5_images = np.concatenate((spider, tseg))
train5_labels = np.concatenate((np.ones(len(spider)), np.zeros(len(tseg))))

train5_ds = tf.data.Dataset.from_tensor_slices((train5_images, train5_labels))
batch_size = 32

train5 = train5_ds.shuffle(buffer_size=len(train5_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train5, validation_data = train5, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: spider v tseg')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: spider v tseg')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train5,verbose=2)

pred5 = model.predict(train5)
pred5_cnn = (pred5 > 0.5).astype("int32")

cm5 = confusion_matrix(train5_labels, pred5_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm5, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: spider v tseg')
plt.show()

accuracy5 = accuracy_score(train5_labels, pred5_cnn)
model_report5 = classification_report(train5_labels, pred5_cnn)
print(f'Model accuracy: {round(accuracy5,4)}')
print('Classification Report: spider v tseg')
print(f'{model_report5}')

train6_images = np.concatenate((tseg, osf))
train6_labels = np.concatenate((np.ones(len(tseg)), np.zeros(len(osf))))

train6_ds = tf.data.Dataset.from_tensor_slices((train6_images, train6_labels))
batch_size = 32

train6 = train6_ds.shuffle(buffer_size=len(train6_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train6, validation_data = train6, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: tseg v osf')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: tseg v osf')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train6,verbose=2)

pred6 = model.predict(train6)
pred6_cnn = (pred6 > 0.5).astype("int32")

cm6 = confusion_matrix(train6_labels, pred6_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm6, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: tseg v osf')
plt.show()

accuracy6 = accuracy_score(train6_labels, pred6_cnn)
model_report6 = classification_report(train6_labels, pred6_cnn)
print(f'Model accuracy: {round(accuracy6,4)}')
print('Classification Report: tseg v osf')
print(f'{model_report6}')

coords_df = pd.read_csv(r'c:\Users\pchri\Downloads\archive (5)\coords_pretrain.csv')
coords_df.head()

import warnings
from plotly import express

warnings.filterwarnings(action='ignore', category=FutureWarning)
express.scatter(data_frame=coords_df.sample(frac=0.25, random_state=4000), x='relative_x', y='relative_y', color='source', hover_name='filename', height=800)

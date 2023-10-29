from keras.src.layers.attention.multi_head_attention import activation
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, ZeroPadding3D,Input, SpatialDropout1D, GlobalMaxPooling3D, BatchNormalization, TimeDistributed, GlobalMaxPooling1D, MaxPooling2D, MaxPooling1D, LSTM, GRU, Reshape, Conv3D, MaxPooling3D, Flatten, Dense, Conv2D, MaxPooling2D, Input, MaxPool3D, BatchNormalization, Dropout
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#load the arrays created in the frame extarction function
segments=np.load('/path-to-save-video/segments.npy/segments.npy', allow_pickle=True)
labels=np.load('/path-to-save-video/labels.npy/labels.npy', allow_pickle=True)
test_segments=np.load('/path-to-save-video/test_segments.npy', allow_pickle=True)
test_labels=np.load('/path-to-save-video/test_labels.npy', allow_pickle=True)

num_segments, frames_per_segment, height, width=segments.shape

#build CNN model
model = Sequential()
model.add(Conv3D(32, (3, 3,3), activation='relu', input_shape=(frames_per_segment, height, width, 1)))
model.add(MaxPooling3D((2, 2,2)))
model.add(Conv3D(64, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D((2, 2,2)))
model.add(Conv3D(64, (3, 3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#output model summary
model.summary()

#complie model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#fit the model
history = model.fit(segments, labels, epochs=10, batch_size=32,
                    validation_split=0.2)


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#evaluate on testing data and print out the test loss and test accuracy 
test_loss, test_acc=model.evaluate(test_segments, test_labels)
print("test loss",test_loss)
print("test accuracy",test_acc)

#run test data on the model to get predictions
predictions=model.predict(test_segments)

#arrange into classes of 0 or 1 (normal or shoplifting)
pred_labels=[]
for i, predicted in enumerate(predictions):
    if predicted[0] > 0.5:
        pred_labels.append(1)
    else:
        pred_labels.append(0)

#show predictions in confusion matrix
cm = confusion_matrix(test_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#create and print classification report
labels = ['Normal', 'Shoplifting']
print(classification_report(test_labels, pred_labels, target_names=labels))

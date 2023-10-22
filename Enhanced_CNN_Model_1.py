import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, ZeroPadding3D, TimeDistributed, MaxPooling2D, MaxPooling1D, LSTM, GRU, Reshape, Conv3D, MaxPooling3D, Flatten, Dense, Conv2D, MaxPooling2D, Input, MaxPool3D, BatchNormalization, Dropout
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import matplotlib.pyplot as plt


segments=np.load('/content/drive/MyDrive/UCF/10_videos/cv_frames/cv_frames_shoplifting/segments.npy', allow_pickle=True)
labels=np.load('/content/drive/MyDrive/UCF/10_videos/cv_frames/cv_frames_shoplifting/labels.npy', allow_pickle=True)
test_segments=np.load('/content/drive/MyDrive/UCF/10_videos/cv_frames/cv_frames_shoplifting/test_segments.npy', allow_pickle=True)
test_labels=np.load('/content/drive/MyDrive/UCF/10_videos/cv_frames/cv_frames_shoplifting/test_labels.npy', allow_pickle=True)


num_segments, frames_per_segment, height, width=segments.shape

#counts the number of unique labels for training and testing 
def count_unique(arr):
  unique, counts = np.unique(arr, return_counts=True)
  return dict(zip(unique,counts))

print("training labels",count_unique(labels))
print("test labels",count_unique(test_labels))


model = Sequential()
model.add(Conv3D(input_shape=(frames_per_segment,height,width,1),filters=16,kernel_size=(3,3,3),padding="same", activation="relu"))
model.add(Conv3D(filters=16,kernel_size=(3,3,3),padding="same", activation="relu"))
model.add(MaxPooling3D(pool_size=(2,2,2), padding='same'))

model.add(Conv3D(filters=32,kernel_size=(3,3,3),padding="same", activation="relu"))
model.add(Conv3D(filters=32,kernel_size=(3,3,3),padding="same", activation="relu"))
model.add(MaxPooling3D(pool_size=(2,2,2), padding='same'))

model.add(Conv3D(filters=64, kernel_size=(3,3,3), padding="same", activation="relu"))
model.add(Conv3D(filters=64, kernel_size=(3,3,3), padding="same", activation="relu"))
model.add(MaxPooling3D(pool_size=(2,2,2),padding='same'))
model.add(Flatten())

model.add(Dense(units=32,activation="relu"))
model.add(Dense(1, activation="sigmoid"))



optimiser=tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    name="Adam",
)


model.compile(optimizer=optimiser,loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

history = model.fit(segments, labels, epochs=10,
                    validation_split=0.2, batch_size=32)

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

test_loss, test_acc=model.evaluate(test_segments, test_labels)
print("loss", test_loss)
print("accuracy", test_acc)



predictions=model.predict(test_segments)

pred_labels=[]
for i, predicted in enumerate(predictions):
    if predicted[0] > 0.5:
        pred_labels.append(1)
    else:
        pred_labels.append(0)

cm = confusion_matrix(test_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

labels = ['Normal', 'Shoplifting']
print(classification_report(test_labels, pred_labels, target_names=labels))

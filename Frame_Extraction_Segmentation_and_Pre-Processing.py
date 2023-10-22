import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, ZeroPadding3D, TimeDistributed, MaxPooling2D, MaxPooling1D, LSTM, GRU, Reshape, Conv3D, MaxPooling3D, Flatten, Dense, Conv2D, MaxPooling2D, Input, MaxPool3D, BatchNormalization, Dropout
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import glob

# Path to your video file
shoplifting_training_videos = '/path_to_my_google_drive'
normal_training_videos = '/path_to_my_google_drive'
shoplifting_testing_videos = '/path_to_my_google_drive'
normal_testing_videos = '/path_to_my_google_drive'

# Opening the video
def segment_creator(directory_path):
    video_files = glob.glob(directory_path + "/*.mp4")
  
    segments_per_video = []
    labels_per_video = []

    segment_length = 5
    fps_to_sample = 5
    for video_file in video_files:
        video = cv2.VideoCapture(video_file)
      
        frame_rate = int(video.get(cv2.CAP_PROP_FPS))

        frame_interval = int(frame_rate/fps_to_sample)

        segments = []
        labels = []

        segment_frames = []
        frame_index = 0
        while True:
            ret, frame = video.read()

            if not ret:
                break
            frame_index += 1

            if frame_index % frame_interval !=0:
                continue
            frame=(frame/255).astype(np.float32)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            segment_frames.append(cv2.resize(frame, (200,160)))


            if len(segment_frames) == segment_length * fps_to_sample:
                segments_per_video.append(segment_frames)

                if "Normal" in video_file:
                    label = 0
                    labels.append(label)
                    labels_per_video.append(label)
                else:
                    label = 1
                    labels.append(label)
                    labels_per_video.append(label)

                segment_frames = []
                frame_index = 0

        video.release()

    # Convert segments and labels to numpy arrays
    segments_per_video_array = np.array(segments_per_video)
    labels_per_video_array = np.array(labels_per_video)


    return segments_per_video_array, labels_per_video_array

# Call the segment_creator function for the desired directories
segments_per_video_1, labels_per_video_1 = segment_creator(shoplifting_training_videos) 
segments_per_video_2, labels_per_video_2 = segment_creator(normal_training_videos)
segments=np.concatenate((segments_per_video_1, segments_per_video_2), axis=0)
labels=np.concatenate((labels_per_video_1, labels_per_video_2), axis=0)

testing_segments_per_video_1, testing_labels_per_video_1 = segment_creator(shoplifting_testing_videos)
testing_segments_per_video_2, testing_labels_per_video_2 = segment_creator(normal_testing_videos)
test_segments=np.concatenate((testing_segments_per_video_1, testing_segments_per_video_2), axis=0)
test_labels=np.concatenate((testing_labels_per_video_1, testing_labels_per_video_2), axis=0)

np.save('/content/drive/MyDrive/UCF/10_videos/cv_frames/cv_frames_shoplifting/segments.npy', segments)
np.save('/content/drive/MyDrive/UCF/10_videos/cv_frames/cv_frames_shoplifting/labels.npy', labels)
np.save('/content/drive/MyDrive/UCF/10_videos/cv_frames/cv_frames_shoplifting/test_segments.npy', test_segments)
np.save('/content/drive/MyDrive/UCF/10_videos/cv_frames/cv_frames_shoplifting/test_labels.npy', test_labels)

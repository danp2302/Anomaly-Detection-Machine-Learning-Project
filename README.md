# Anomaly-Detection-Machine-Learning-Project
A project which classifies video segments as 'normal' or 'anomalous' (shoplifting).

To run the code, you first of all need to go to the frame extraction and segmentation file. 
Update the path directory for the videos so it matches your storage, the path takes in mp4 videos not individual frames so the videos need to be inside these folders

shoplifting_training_videos = '/path_to_my_google_drive'
normal_training_videos = '/path_to_my_google_drive'
shoplifting_testing_videos = '/path_to_my_google_drive'
normal_testing_videos = '/path_to_my_google_drive'

Run the code and wait for it do the frame extraction/segmentation/pre-processing. 

Once finished you can than run each individual model file in python and wait for it to finish and output the results.
I created this in jupiter noteboks on google colab so you can also move the code over to that platform.

import os
import pickle
import tensorflow as tf
from django.shortcuts import render
from moviepy.editor import VideoFileClip
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-defined tokenizer from a .pkl file
tokenizer_path = os.path.join('detector', 'models', 'tokenizer.pkl')
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

model_spam = tf.keras.models.load_model(os.path.join('detector', 'Models', 'spam_ham_model.h5'))
model_voice = tf.keras.models.load_model(os.path.join('detector', 'Models', 'MFCCs 85.h5'))

def index(request):
    if request.method == 'POST':
        # Handle email text submission
        if 'email_text' in request.POST:
            email_text = request.POST['email_text']
            # Tokenize and pad the sequences
            new_email_sequences = tokenizer.texts_to_sequences([email_text])
            maxlen = 100  # Ensure this matches the maxlen used during training
            new_email_padded = pad_sequences(new_email_sequences, padding='post', maxlen=maxlen)

            # Make a prediction
            prediction = model_spam.predict(new_email_padded)

            # Set the threshold and classify
            threshold = 0.5
            if prediction[0] > threshold:
                email_result = "This email is spam."
            else:
                email_result = "This email is not spam."

        # Handle video file upload
        elif 'video_file' in request.FILES:
            video_file = request.FILES['video_file']
            video_path = f'temp_video.mp4'

            # Save the uploaded video to a temporary location
            with open(video_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)

            # Process the video file
            audio_output_path = 'extracted_audio.wav'
            extract_audio_from_video(video_path, audio_output_path)

            # Change sample rate
            resampled_audio, _ = resample_audio(audio_output_path)

            # Split audio into segments
            audio_segments = split_audio_into_segments(resampled_audio, 32000)

            # Generate MFCCs
            mfcc_segments = [create_mfcc_from_segment(seg) for seg in audio_segments]

            # Scale the MFCC segments
            scaler = StandardScaler()
            scaled_mfcc_segments = [scaler.fit_transform(segment) for segment in mfcc_segments]

            # Make predictions on each segment
            predictions = []
            for segment in mfcc_segments:
                segment = segment.reshape(1, 13, 20)
                prediction = model_voice.predict(segment)
                predictions.append(float(prediction))

            # Determine if it's a deepfake
            det = sum(1 for p in predictions if p <= 0.5)
            if det >= 1:
                video_result = "It may be a deepfake."
            else:
                video_result = "It is not a deepfake."

    return render(request, 'detector/index.html', {
        'email_result': locals().get('email_result', ''),
        'video_result': locals().get('video_result', ''),
    })

def extract_audio_from_video(input_file, output_file):
    # Use moviepy to extract audio
    with VideoFileClip(input_file) as video:
        video.audio.write_audiofile(output_file)

def resample_audio(audio_path, target_sr=16000):
    y, sr = librosa.load(audio_path, sr=None)  # Load with original sampling rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr

def split_audio_into_segments(audio, segment_length):
    num_segments = len(audio) // segment_length
    segments = np.array_split(audio[:num_segments * segment_length], num_segments)
    return segments

def create_mfcc_from_segment(segment, sr=16000, n_mfcc=13, n_mels=20):
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels)
    return mfccs[:, :n_mels]  # Ensure it has 20 time steps


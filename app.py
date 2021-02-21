from flask import Flask, redirect, url_for,session,request,render_template,session,flash
import time
import os
import numpy as np

## Audio Preprocessing ##
import pyaudio
import wave
import librosa
from scipy.stats import zscore
frames=[]
app=Flask(__name__)
app.secret_key='abc'
'''
Mel-spectogram computation
'''
def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
  # Compute spectogram
  mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

  # Compute mel spectrogram
  mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)

  # Compute log-mel spectrogram
  mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

  return np.asarray(mel_spect)


'''
Audio framing
'''
def frame(y, win_step=64, win_size=128):

    # Number of frames
    nb_frames = 1 + int((y.shape[2] - win_size) / win_step)

    # Framming
    frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size)).astype(np.float16)
    for t in range(nb_frames):
        frames[:,t,:,:] = np.copy(y[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float16)

    return frames
'''
Predict speech emotion over time from an audio file
'''
max_pad_len = 49100
def predict_emotion_from_file(filename, chunk_step=16000, chunk_size=49100, predict_proba=False, sample_rate=16000):

  # Read audio file
  y, sr = librosa.core.load(filename, sr=sample_rate, offset=0.5)
  # Padding or truncated signal
  print(y,"Hello")
  if len(y) < max_pad_len:
    y_padded = np.zeros(max_pad_len)
    y_padded[:len(y)] = y
    y = y_padded
  elif len(y) > max_pad_len:
    y = np.asarray(y[:max_pad_len])
    # Split audio signals into chunks
  chunks = frame(y.reshape(1, 1, -1), chunk_step, chunk_size)
  # Reshape chunks
  chunks = chunks.reshape(chunks.shape[1],chunks.shape[-1])
  # Z-normalization
  y = np.asarray(list(map(zscore, chunks)))
  # Compute mel spectrogram
  mel_spect = np.asarray(list(map(mel_spectrogram, y)))
  # Time distributed Framing
  mel_spect_ts = frame(mel_spect)
  print(mel_spect_ts,"Hello")
  # Build X for time distributed CNN
  X = mel_spect_ts.reshape(mel_spect_ts.shape[0],
                                mel_spect_ts.shape[1],
                                mel_spect_ts.shape[2],
                                mel_spect_ts.shape[3],
                                1)
  # print(X)
  return X
@app.route('/',methods=['GET','POST'])
def home():
    # Voice Record sub dir
    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    a=predict_emotion_from_file('C:/Users/KARTHIK SURINENI/OneDrive/Desktop/Speech Emotion/03-01-01-01-01-01-01.wav', chunk_step=step*sample_rate)
    # print(a)
    return render_template('index.html',a=a.tolist())
if __name__=='__main__':
	app.run(debug=True)

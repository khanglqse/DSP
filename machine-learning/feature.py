import random
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Tải dữ liệu và lọc hai loại âm thanh
esc50_path = './meta'
audio_folder = './audio'
data = pd.read_csv(esc50_path + '/metadata.csv')

# Tạo danh sách các đường dẫn đầy đủ cho các tệp thuộc loại 'dog'
dog_bark_files = [os.path.join(audio_folder, filename) for filename in data[data['category'] == 'dog']['filename']]
sea_wave_files = [os.path.join(audio_folder, filename) for filename in data[data['category'] == 'sea_waves']['filename']]
cat_wave_files = [os.path.join(audio_folder, filename) for filename in data[data['category'] == 'cat']['filename']]
rain_wave_files = [os.path.join(audio_folder, filename) for filename in data[data['category'] == 'rain']['filename']]
water_wave_files = [os.path.join(audio_folder, filename) for filename in data[data['category'] == 'pouring_water']['filename']]
wind_wave_files = [os.path.join(audio_folder, filename) for filename in data[data['category'] == 'wind']['filename']]

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_features_comparison(file_paths1, label1, file_paths2, label2):
    # Thiết lập thông số cho các hàm tách feature
    feature_params = {
        'n_fft': 2048,
        'hop_length': 512
    }
    
    # Tạo figure cho các biểu đồ so sánh
    fig, axs = plt.subplots(3, 4, figsize=(18, 12))
    
    # Dạng sóng theo miền thời gian cho từng file
    for i, file_path in enumerate(file_paths1):
        y, sr = librosa.load(file_path)
        axs[0, 0].plot(np.linspace(0, len(y) / sr, len(y)), y, label=f"{label1} #{i+1}")
    axs[0, 0].set_title(f"Waveform of {label1} (5 samples)")
    axs[0, 0].set_ylabel("Amplitude")
    
    for i, file_path in enumerate(file_paths2):
        y, sr = librosa.load(file_path)
        axs[0, 1].plot(np.linspace(0, len(y) / sr, len(y)), y, label=f"{label2} #{i+1}")
    axs[0, 1].set_title(f"Waveform of {label2} (5 samples)")
    axs[0, 1].set_ylabel("Amplitude")

    # Spectral Roll-off
    for file_path in file_paths1:
        y, sr = librosa.load(file_path)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'])
        axs[0, 2].plot(rolloff[0], label=label1)
    for file_path in file_paths2:
        y, sr = librosa.load(file_path)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'])
        axs[0, 3].plot(rolloff[0], label=label2)
    axs[0, 2].set_title("Spectral Roll-off Comparison for {}".format(label1))
    axs[0, 3].set_title("Spectral Roll-off Comparison for {}".format(label2))

    # Spectral Centroid
    for file_path in file_paths1:
        y, sr = librosa.load(file_path)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'])
        axs[1, 0].plot(centroid[0], label=label1)
    for file_path in file_paths2:
        y, sr = librosa.load(file_path)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'])
        axs[1, 1].plot(centroid[0], label=label2)
    axs[1, 0].set_title("Spectral Centroid Comparison for {}".format(label1))
    axs[1, 1].set_title("Spectral Centroid Comparison for {}".format(label2))
    
    # Zero Crossing Rate
    for file_path in file_paths1:
        y, sr = librosa.load(file_path)
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=feature_params['n_fft'], hop_length=feature_params['hop_length'])
        axs[1, 2].plot(zcr[0], label=label1)
    for file_path in file_paths2:
        y, sr = librosa.load(file_path)
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=feature_params['n_fft'], hop_length=feature_params['hop_length'])
        axs[1, 3].plot(zcr[0], label=label2)
    axs[1, 2].set_title("Zero Crossing Rate for {}".format(label1))
    axs[1, 3].set_title("Zero Crossing Rate for {}".format(label2))
    
    # MFCC
    for file_path in file_paths1:
        y, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'], n_mfcc=13)
        img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=feature_params['hop_length'], ax=axs[2, 0], cmap='viridis')
    axs[2, 0].set_title("MFCC for {}".format(label1))
    fig.colorbar(img, ax=axs[2, 0], format="%+2.f dB")
    
    for file_path in file_paths2:
        y, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'], n_mfcc=13)
        img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=feature_params['hop_length'], ax=axs[2, 1], cmap='viridis')
    axs[2, 1].set_title("MFCC for {}".format(label2))
    fig.colorbar(img, ax=axs[2, 1], format="%+2.f dB")

    # Chroma Feature
    for file_path in file_paths1:
        y, sr = librosa.load(file_path)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'])
        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, hop_length=feature_params['hop_length'], ax=axs[2, 2], cmap='cool')
    axs[2, 2].set_title("Chroma for {}".format(label1))
    fig.colorbar(img, ax=axs[2, 2])

    for file_path in file_paths2:
        y, sr = librosa.load(file_path)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'])
        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, hop_length=feature_params['hop_length'], ax=axs[2, 3], cmap='cool')
    axs[2, 3].set_title("Chroma for {}".format(label2))
    fig.colorbar(img, ax=axs[2, 3])

    # Điều chỉnh bố cục biểu đồ
    plt.tight_layout()
    plt.show()


random.seed(3)
plot_features_comparison(random.sample(dog_bark_files, 5), 'Dog', random.sample(cat_wave_files, 5), 'Cat')

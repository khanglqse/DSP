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
import numpy as np
import matplotlib.pyplot as plt

def plot_features_comparison(file_path1, label1, file_path2, label2):
    # Thiết lập thông số cho các hàm tách feature
    feature_params = {
        'n_fft': 2048,
        'hop_length': 512
    }
    
    # Load âm thanh
    y1, sr1 = librosa.load(file_path1)
    y2, sr2 = librosa.load(file_path2)
    
    # Trích xuất MFCC
    

    # Vẽ biểu đồ so sánh
    fig, axs = plt.subplots(3, 2, figsize=(10, 14))
    
    # So sánh biên độ (amplitude) - Dạng sóng theo miền thời gian
    axs[0, 0].plot(np.linspace(0, len(y1)/sr1, len(y1)), y1, color='blue', label=label1)
    axs[0, 0].set_title("Waveform of {}".format(label1))
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].legend()
    
    axs[0, 1].plot(np.linspace(0, len(y2)/sr2, len(y2)), y2, color='red', label=label2)
    axs[0, 1].set_title("Waveform of {}".format(label2))
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].legend()
    rolloff1 = librosa.feature.spectral_rolloff(y=y1, sr=sr1, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'])
    rolloff2 = librosa.feature.spectral_rolloff(y=y2, sr=sr2, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'])

    axs[1, 0].plot(rolloff1[0], color='blue', label=label1)
    axs[1, 0].plot(rolloff2[0], color='red', label=label1)
    axs[1, 0].set_title("Spectral Roll-off Comparison ({}".format(label1))
    axs[1, 0].set_xlabel("Frames")
    axs[1, 0].set_ylabel("Roll-off Frequency (Hz)")
    
    
    # Các biểu đồ khác như Spectral Centroid, Chroma và Zero Crossing Rate có thể giữ nguyên
    spectral_centroid1 = librosa.feature.spectral_centroid(y=y1, sr=sr1, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'])
    spectral_centroid2 = librosa.feature.spectral_centroid(y=y2, sr=sr2, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'])

    # So sánh Spectral Centroid
    axs[2, 0].plot(spectral_centroid1[0], color='blue', label=label1)
    axs[2, 0].plot(spectral_centroid2[0], color='red', label=label2)
    axs[2, 0].set_title("Spectral Centroid Comparison")
    axs[2, 0].legend()

    # So sánh Zero Crossing Rate
    zcr1 = librosa.feature.zero_crossing_rate(y1, frame_length=2048, hop_length=feature_params['hop_length'])
    zcr2 = librosa.feature.zero_crossing_rate(y2, frame_length=2048, hop_length=feature_params['hop_length'])
    axs[2, 1].plot(zcr1[0], color='blue', label=label1)
    axs[2, 1].plot(zcr2[0], color='red', label=label2)
    axs[2, 1].set_title("Zero Crossing Rate Comparison")
    axs[2, 1].set_xlabel("Frames")
    axs[2, 1].set_ylabel("Zero Crossing Rate")
    axs[2, 1].legend()
    
  
    chroma1 = librosa.feature.chroma_stft(y=y1, sr=sr1, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'])
    chroma2 = librosa.feature.chroma_stft(y=y2, sr=sr2, n_fft=feature_params['n_fft'], hop_length=feature_params['hop_length'])
    axs[1, 1].imshow(chroma1, aspect='auto', origin='lower', cmap='jet', alpha=0.5, label=label1)
    axs[1, 1].imshow(chroma2, aspect='auto', origin='lower', cmap='coolwarm', alpha=0.5, label=label2)
    axs[1, 1].set_title("Chroma Comparison")
    # Điều chỉnh bố cục biểu đồ
    plt.tight_layout()
    plt.show()


random.seed(3);
plot_features_comparison(random.choice(cat_wave_files) , 'Dog Bark', random.choice(rain_wave_files), 'Wind')
  
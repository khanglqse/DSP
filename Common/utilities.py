import librosa
import numpy as np

def extract_features(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

        # Trích xuất MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # Trích xuất Spectrogram
        spectrogram = np.abs(librosa.stft(audio_data))
        spectrogram_scaled = np.mean(spectrogram, axis=1)
        
        # Trích xuất Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
        mel_spectrogram_scaled = np.mean(mel_spectrogram.T, axis=0)

        # Trích xuất Zero-Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)
        zcr_scaled = np.mean(zero_crossing_rate.T, axis=0)

        # Trích xuất RMS Energy
        rms = librosa.feature.rms(y=audio_data)
        rms_scaled = np.mean(rms.T, axis=0)

        # Kết hợp tất cả các đặc trưng
        features_combined = np.hstack([mfccs_scaled, spectrogram_scaled, mel_spectrogram_scaled, zcr_scaled, rms_scaled])
        return features_combined
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}, error: {e}")
        return None

import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
audio_path = 'path_to_audio_file.wav'
audio_data, sample_rate = librosa.load(audio_path, sr=None)

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)

# Plot MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
plt.colorbar()
plt.title('MFCCs')
plt.tight_layout()
plt.show()

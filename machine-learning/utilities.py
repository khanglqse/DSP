import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from joblib import dump, load

data_dir = './audio'
original_metadata_file = './meta/metadata.csv'
augmented_metadata_file = './meta/augmented_metadata.csv'
augmented_data_dir = './augmented_audio'
os.makedirs(augmented_data_dir, exist_ok=True)

def augment_audio(file_path, label, fold, category, esc10, src_file, take):
    y, sr = librosa.load(file_path)

    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    pitch_file = os.path.join(augmented_data_dir, 'pitch_' + os.path.basename(file_path))
    sf.write(pitch_file, y_pitch, sr)

    y_speed = librosa.effects.time_stretch(y, rate=1.5)
    speed_file = os.path.join(augmented_data_dir, 'speed_' + os.path.basename(file_path))
    sf.write(speed_file, y_speed, sr)

    noise = np.random.normal(0, 0.005, y.shape)
    y_noise = y + noise
    noise_file = os.path.join(augmented_data_dir, 'noise_' + os.path.basename(file_path))
    sf.write(noise_file, y_noise, sr)

    return [
        (os.path.basename(pitch_file), fold, label, category, esc10, src_file, take),
        (os.path.basename(speed_file), fold, label, category, esc10, src_file, take),
        (os.path.basename(noise_file), fold, label, category, esc10, src_file, take),
    ]

def update_metadata(original_metadata_file, augmented_files, augmented_metadata_file):
    # Load original metadata
    metadata = pd.read_csv(original_metadata_file)

    # Create DataFrame for augmented data
    augmented_df = pd.DataFrame(augmented_files, columns=['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take'])

    # Combine original metadata with augmented data
    combined_metadata = pd.concat([metadata, augmented_df], ignore_index=True)

    # Save the combined metadata to a new CSV file
    combined_metadata.to_csv(augmented_metadata_file, index=False)

def apply_augment(original_metadata_file, augmented_metadata_file):
    metadata = pd.read_csv(original_metadata_file)
    augmented_files = []

    for index, row in metadata.iterrows():
        file_path = os.path.join(data_dir, row['filename'])  
        fold = row['fold']
        target = row['target']
        category = row['category']
        esc10 = row['esc10']
        take = row['take']
        if category in ['drinking_sipping', 'fireworks', 'mouse_click']:
            augmented_files.extend(augment_audio(file_path, target, fold, category, esc10, src_file=row['src_file'], take=take))  

    update_metadata(original_metadata_file, augmented_files, augmented_metadata_file) 

# Usage

def load_preprocessed_data():
    X = load('X_features_init.joblib')
    y = load('y_labels_init.joblib')
    le = load('label_encoder_init.joblib')
    return X, y, le
# apply_augment(os.path.join('./meta','metadata.csv'))
apply_augment(original_metadata_file, augmented_metadata_file)

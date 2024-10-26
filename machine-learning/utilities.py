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

    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5)
    pitch_file = os.path.join(augmented_data_dir, 'pitchv2_' + os.path.basename(file_path))
    sf.write(pitch_file, y_pitch, sr)

    y_speed = librosa.effects.time_stretch(y, rate=1.2)
    speed_file = os.path.join(augmented_data_dir, 'speedv2_' + os.path.basename(file_path))
    sf.write(speed_file, y_speed, sr)

    noise = np.random.normal(0, 0.0025, y.shape)
    y_noise = y + noise
    noise_file = os.path.join(augmented_data_dir, 'noisev2_' + os.path.basename(file_path))
    sf.write(noise_file, y_noise, sr)

    y_louder = librosa.util.normalize(y * 1.25)
    louder_file = os.path.join(augmented_data_dir, 'louderv2_' + os.path.basename(file_path))
    sf.write(louder_file, y_louder, sr)
    return [
        (os.path.basename(pitch_file), fold, label, category, esc10, src_file, take),
        (os.path.basename(speed_file), fold, label, category, esc10, src_file, take),
        (os.path.basename(noise_file), fold, label, category, esc10, src_file, take),
        (os.path.basename(louder_file), fold, label, category, esc10, src_file, take),
    ]

def update_metadata(metadata_file, augmented_files):
    metadata = pd.read_csv(metadata_file)

    for augmented_file in augmented_files:
        new_entry = pd.DataFrame({
            'filename': [augmented_file[0]],
            'fold': [augmented_file[1]],
            'target': [augmented_file[2]],
            'category': [augmented_file[3]],
            'esc10': [augmented_file[4]],
            'src_file': [augmented_file[5]],
            'take': [augmented_file[6]],
        })
        metadata = pd.concat([metadata, new_entry], ignore_index=True)

    metadata.to_csv(metadata_file, index=False)
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
        if category in ['mouse_click', 'breathing']:
            augmented_files.extend(augment_audio(file_path, target, fold, category, esc10, src_file=row['src_file'], take=take))  

    update_metadata(augmented_metadata_file, augmented_files) 

# Usage

def load_preprocessed_data():
    X = load('X_features_init.joblib')
    y = load('y_labels_init.joblib')
    le = load('label_encoder_init.joblib')
    return X, y, le

def load_preprocessed_aug_data():
    X = load('X_features_aug.joblib')
    y = load('y_labels_aug.joblib')
    le = load('label_encoder_aug.joblib')
    return X, y, le
def load_preprocessed_final_data():
    X = load('X_final_v2.joblib')
    y = load('y_final_v2.joblib')
    le = load('label_encoder_v2.joblib')
    return X, y, le

def load_preprocessed_final_v1_data():
    X = load('X_final.joblib')
    y = load('y_final.joblib')
    le = load('label_encoder.joblib')
    return X, y, le
# apply_augment(original_metadata_file, augmented_metadata_file)

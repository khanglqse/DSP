import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd

data_dir = './audio'
os.makedirs(data_dir, exist_ok=True)

def augment_audio(file_path, label, fold, category, esc10, src_file, take):
    y, sr = librosa.load(file_path)

    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    pitch_file = os.path.join(data_dir, 'pitch_' + os.path.basename(file_path))
    sf.write(pitch_file, y_pitch, sr)

    y_speed = librosa.effects.time_stretch(y, rate=1.5)
    speed_file = os.path.join(data_dir, 'speed_' + os.path.basename(file_path))
    sf.write(speed_file, y_speed, sr)

    noise = np.random.normal(0, 0.005, y.shape)
    y_noise = y + noise
    noise_file = os.path.join(data_dir, 'noise_' + os.path.basename(file_path))
    sf.write(noise_file, y_noise, sr)

    return [
        (os.path.basename(pitch_file), fold, label, category, esc10, src_file, take),
        (os.path.basename(speed_file), fold, label, category, esc10, src_file, take),
        (os.path.basename(noise_file), fold, label, category, esc10, src_file, take),
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

def apply_augment(metadata_file):
    metadata = pd.read_csv(metadata_file)
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

    update_metadata(metadata_file, augmented_files) 

apply_augment(os.path.join('./meta','metadata.csv'))

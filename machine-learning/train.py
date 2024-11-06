import os
import librosa
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.svm import SVC
from joblib import dump, load
import utilities

metadata_path = './meta/metadata.csv'
augmented_metadata_path = './meta/augmented_metadata.csv'
audio_folder = './audio'
augmented_audio_folder = './augmented_audio'

metadata = pd.read_csv(metadata_path)
augmented_metadata = pd.read_csv(augmented_metadata_path)

def extract_features(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        chroma_scaled = np.mean(chroma.T, axis=0)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
        spectral_contrast_scaled = np.mean(spectral_contrast.T, axis=0)

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)
        zero_crossing_rate_scaled = np.mean(zero_crossing_rate.T, axis=0)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        spectral_centroid_scaled = np.mean(spectral_centroid.T, axis=0)

        features = np.hstack([
            mfccs_scaled,
            chroma_scaled,
            spectral_contrast_scaled,
            zero_crossing_rate_scaled,
            spectral_centroid_scaled
        ])
        
        return features
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}, error: {e}")
        return None

def pre_processing():
    features = Parallel(n_jobs=-1)(delayed(extract_features)(os.path.join(audio_folder, row['filename'])) for _, row in metadata.iterrows())
    augmented_features = Parallel(n_jobs=-1)(delayed(extract_features)(os.path.join(augmented_audio_folder, row['filename'])) for _, row in augmented_metadata.iterrows())

    features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features[0].shape[0])])
    features_df['class_label'] = metadata['category']
    features_df = features_df.dropna()

    augmented_features_df = pd.DataFrame(augmented_features, columns=[f'feature_{i}' for i in range(augmented_features[0].shape[0])])
    augmented_features_df['class_label'] = augmented_metadata['category']  # Sử dụng category từ augmented_metadata
    augmented_features_df = augmented_features_df.dropna()

    combined_df = pd.concat([features_df, augmented_features_df], ignore_index=True)

    X = np.array(combined_df.iloc[:, :-1])
    y = np.array(combined_df['class_label'])
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    # dump(X, 'X_final_v2.joblib')
    # dump(y, 'y_final_v2.joblib')
    # dump(le, 'label_encoder_v2.joblib')
    return X, y, le
  


def show_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                cbar_kws={'label': 'Frequency'})
    
    plt.ylabel('Actual', fontsize=8)
    plt.xlabel('Predicted', fontsize=8)
    plt.title('Confusion Matrix', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    for text in plt.gca().texts:
        text.set_fontsize(8)  

    plt.show()

def random_forest():
    # X, y, le = pre_processing()
    X, y, le = utilities.load_preprocessed_data()
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_

    y_val_pred = best_model.predict(X_val)
    print("Validation Set Evaluation")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred)}")
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))
    show_confusion_matrix(y_val, y_val_pred, le.classes_, normalize=True)
    
    # # Step 5: Final evaluation on the test set
    # y_test_pred = best_model.predict(X_test)
    # print("Test Set Evaluation")
    # print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")
    # print(classification_report(y_test, y_test_pred, target_names=le.classes_))
    # show_confusion_matrix(y_test, y_test_pred, le.classes_, normalize=True)


def SVM():
    X, y, le = utilities.load_preprocessed_final_data()
  
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    model = SVC(random_state=42)
    param_grid = {
        'C': [100],
        'kernel': ['rbf'],
        'gamma': ['scale']
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best Hyperparameters: ", grid_search.best_params_)
    dump(best_model, 'svm_model.joblib')
    y_val_pred = best_model.predict(X_val) 

    print("Validation Set Evaluation")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred)}")
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))
    show_confusion_matrix(y_val, y_val_pred, le.classes_, normalize=True)
    
    y_test_pred = best_model.predict(X_test)
    print("Test Set Evaluation")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))
    show_confusion_matrix(y_test, y_test_pred, le.classes_, normalize=True)

def SVM1():
    X, y, le = utilities.load_preprocessed_data()
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = SVC(C=1, kernel='rbf', gamma='scale', random_state=42)  # Adjust these values as needed

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)

    print("Validation Set Evaluation")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred)}")
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))
    show_confusion_matrix(y_val, y_val_pred, le.classes_, normalize=True)


SVM()
# random_forest()
# pre_processing()
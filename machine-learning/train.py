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

# Đường dẫn đến file metadata và thư mục audio
metadata_path = './meta/metadata.csv'
audio_folder = './audio'

# Đọc file metadata
metadata = pd.read_csv(metadata_path)

# Hàm trích xuất đặc trưng âm thanh từ các file audio
def extract_features(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        chroma_scaled = np.mean(chroma.T, axis=0)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
        spectral_contrast_scaled = np.mean(spectral_contrast.T, axis=0)
        return np.hstack([mfccs_scaled, chroma_scaled, spectral_contrast_scaled])
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}, error: {e}")
        return None

def pre_processing():
    features = Parallel(n_jobs=-1)(delayed(extract_features)(os.path.join(audio_folder, row['filename'])) for _, row in metadata.iterrows())
    features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features[0].shape[0])])
    features_df['class_label'] = metadata['category']
    features_df = features_df.dropna()
    X = np.array(features_df.iloc[:, :-1])
    y = np.array(features_df['class_label'])
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Handle class imbalance
    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_resample(X, y)

    return X, y, le

def show_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def random_forest():
    X, y, le = pre_processing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
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
    y_pred = best_model.predict(X_test)

    print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    show_confusion_matrix(y_test, y_pred, le.classes_, normalize=True)

def SVM():
    X, y, le = pre_processing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tuning Hyperparameters for SVM
    model = SVC(random_state=42)
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"SVM Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    show_confusion_matrix(y_test, y_pred, le.classes_, normalize=True)

# Uncomment the model you want to run
# random_forest()
SVM()

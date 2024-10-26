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

def pre_processing(audio_folder = audio_folder, metadata = metadata):
    features = Parallel(n_jobs=-1)(delayed(extract_features)(os.path.join(audio_folder, row['filename'])) for _, row in metadata.iterrows())
    features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features[0].shape[0])])
    features_df['class_label'] = metadata['category']
    features_df = features_df.dropna()
    X = np.array(features_df.iloc[:, :-1])
    y = np.array(features_df['class_label'])
    le = LabelEncoder()
    y = le.fit_transform(y)
   

    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_resample(X, y)
    return X, y, le


def show_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    
    plt.figure(figsize=(12, 10))  # Increase figure size
    sns.heatmap(conf_matrix, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                cbar_kws={'label': 'Frequency'})  # Add color bar label
    
    plt.ylabel('Actual', fontsize=8)  # Increase font size for labels
    plt.xlabel('Predicted', fontsize=8)  # Increase font size for labels
    plt.title('Confusion Matrix', fontsize=8)  # Increase font size for title
    plt.xticks(fontsize=8)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=8)  # Increase font size for y-axis ticks
    
    # Adjust annotation font size
    for text in plt.gca().texts:
        text.set_fontsize(8)  # Change annotation font size

    plt.show()

def random_forest():
    # X, y, le = pre_processing()
    X, y, le = utilities.load_preprocessed_data()
    
    # Step 1: Split into training+validation and test sets (80% training+validation, 20% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 2: Split training+validation set into separate training and validation sets (75% train, 25% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Define the model and hyperparameters for tuning
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # Step 3: Tune and train the model with the validation set using GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Select the best model
    best_model = grid_search.best_estimator_

    # Step 4: Evaluate on the validation set to check performance
    y_val_pred = best_model.predict(X_val)
    print("Validation Set Evaluation")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred)}")
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))
    show_confusion_matrix(y_val, y_val_pred, le.classes_, normalize=True)
    
    # Step 5: Final evaluation on the test set
    y_test_pred = best_model.predict(X_test)
    print("Test Set Evaluation")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))
    show_confusion_matrix(y_test, y_test_pred, le.classes_, normalize=True)


def SVM():
    # X, y, le = utilities.load_preprocessed_data()
    X, y, le = utilities.load_preprocessed_data()
    X_aug, y_aug, y_le = pre_processing(augmented_audio_folder, augmented_metadata)
    X_combined = np.concatenate((X, X_aug), axis=0)
    y_combined = np.concatenate((y, y_aug), axis=0)
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_combined, y_combined)
    
    # Step 1: Split into training+validation and test sets (80% training+validation, 20% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Step 2: Split training+validation set into separate training and validation sets (75% train, 25% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    model = SVC(random_state=42)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best Hyperparameters: ", grid_search.best_params_)

    y_val_pred = best_model.predict(X_val) 

    print("Validation Set Evaluation")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred)}")
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))
    show_confusion_matrix(y_val, y_val_pred, le.classes_, normalize=True)
    
    # Step 5: Final evaluation on the test set
    # y_test_pred = best_model.predict(X_test)
    # print("Test Set Evaluation")
    # print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")
    # print(classification_report(y_test, y_test_pred, target_names=le.classes_))
    # show_confusion_matrix(y_test, y_test_pred, le.classes_, normalize=True)

# pre_processing()
def SVM1():
    # Load preprocessed data
    X, y, le = utilities.load_preprocessed_data()
    
    # Step 1: Split into training+validation and test sets (80% training+validation, 20% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 2: Split training+validation set into separate training and validation sets (75% train, 25% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Initialize SVM model with manually set hyperparameters
    model = SVC(C=1, kernel='rbf', gamma='scale', random_state=42)  # Adjust these values as needed

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on validation set
    y_val_pred = model.predict(X_val)

    # Evaluate on the validation set
    print("Validation Set Evaluation")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred)}")
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))
    show_confusion_matrix(y_val, y_val_pred, le.classes_, normalize=True)

SVM()
# random_forest()
# pre_processing()
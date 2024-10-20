# import librosa
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.multioutput import MultiOutputClassifier

# # Load file audio
# audio_file = 'test_audio.wav'
# audio_data, sr = librosa.load(audio_file, sr=None)

# # Chia nhỏ file audio thành các frame dài 2 giây
# frame_length = int(sr * 2)
# hop_length = frame_length
# frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)

# # Hàm trích xuất MFCCs
# def extract_features(audio_frame, sr):
#     mfccs = librosa.feature.mfcc(y=audio_frame, sr=sr, n_mfcc=13)
#     mfccs_scaled = np.mean(mfccs.T, axis=0)
#     return mfccs_scaled

# # Trích xuất đặc trưng cho từng frame
# features = np.array([extract_features(frame, sr) for frame in frames.T])

# # Load hoặc huấn luyện mô hình phân loại (giả sử mô hình đã được huấn luyện)
# model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
# model.fit(X_train, y_train)  # Huấn luyện mô hình với tập dữ liệu có nhãn

# # Dự đoán âm thanh có mặt trong file
# predictions = model.predict(features)

# # Tổng hợp kết quả
# def aggregate_predictions(predictions):
#     unique_labels = set()
#     for pred in predictions:
#         for label_idx, value in enumerate(pred):
#             if value == 1:
#                 unique_labels.add(label_idx)
#     return unique_labels

# labels_present = aggregate_predictions(predictions)
# print(f"Các loại âm thanh có mặt trong file audio: {labels_present}")

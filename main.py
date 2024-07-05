import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load and preprocess images from the dataset directory
def load_images_from_folder(folder_path):
    images = []
    labels = []
    for main_folder in os.listdir(folder_path):
        main_folder_path = os.path.join(folder_path, main_folder)
        if os.path.isdir(main_folder_path):
            for sub_folder in os.listdir(main_folder_path):
                sub_folder_path = os.path.join(main_folder_path, sub_folder)
                if os.path.isdir(sub_folder_path):
                    for image_filename in os.listdir(sub_folder_path):
                        image_path = os.path.join(sub_folder_path, image_filename)
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            image = cv2.resize(image, (64, 64))
                            images.append(image)
                            labels.append(sub_folder)  # Use sub-folder name as label
                        else:
                            print(f'Warning: Unable to read image {image_path}')
    return np.array(images), np.array(labels)

# Path to the dataset folder
dataset_path = 'leapGestRecog'  # Update this with the correct path

# Load images and labels
images, labels = load_images_from_folder(dataset_path)

# Extract HOG features
def extract_hog_features(images):
    hog_features = []
    for image in images:
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

hog_features = extract_hog_features(images)

# Debugging: Print the shape of the HOG features and labels
print(f'HOG features shape: {hog_features.shape}')
print(f'Labels shape: {labels.shape}')
print(f'Number of HOG features: {len(hog_features)}')
print(f'Number of labels: {len(labels)}')

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Check consistency in length
assert len(hog_features) == len(labels_encoded), "Mismatch between number of HOG features and labels"

# Split data
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels_encoded, test_size=0.2, random_state=42)

# Debugging: Print the shapes of the split data
print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

# Train an SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy:.2f}')

# Real-time hand gesture recognition
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    features = hog(resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    features = preprocess_frame(frame).reshape(1, -1)
    
    # Debugging: Print the shape of the features extracted from the frame
    print(f'Features shape (real-time): {features.shape}')
    
    # Predict gesture
    prediction = classifier.predict(features)
    predicted_class = label_encoder.inverse_transform(prediction)

    # Display result
    cv2.putText(frame, f'Gesture: {predicted_class[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

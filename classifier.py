import os
import pickle
import cv2
import numpy as np
from scipy.spatial.distance import cdist

# Path ke model
BOW_FILE_PICKLE = "model/sift_bow_dictionary.pkl"
SCALER_FILE_PICKLE = "model/sift_SCALER_WS_FILE_PICKLE_SIFT_full.pkl"
SVM_FILE_PICKLE = "model/sift_svm_with_sift_model_claude_full.pkl"

# Load Pickle File
def load_file_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def center_handwritten_image(image, canvas_size=(192, 192)):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cropped = gray[y:y+h, x:x+w]
    canvas = np.ones(canvas_size, dtype=np.uint8) * 255
    start_x = max((canvas_size[1] - w) // 2, 0)
    start_y = max((canvas_size[0] - h) // 2, 0)
    canvas[start_y:start_y+h, start_x:start_x+w] = cropped
    return canvas

def preprocess_image(filepath):
    image = cv2.imread(filepath)
    img = cv2.resize(image, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    centered = center_handwritten_image(img)
    equalized = cv2.equalizeHist(centered)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(equalized, None)
    img_with_keypoints = cv2.drawKeypoints(equalized, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    original_name = os.path.splitext(os.path.basename(filepath))[0]
    preprocessed_path = os.path.join("static/results", f"processed_image_{original_name}.jpg")
    cv2.imwrite(preprocessed_path, img_with_keypoints)
    return preprocessed_path, equalized

def create_feature_bow(image_descriptor, bow, num_cluster):
    features = np.zeros(num_cluster, dtype=float)
    if image_descriptor is not None:
        distance = cdist(image_descriptor, bow)
        argmin = np.argmin(distance, axis=1)
        for j in argmin:
            features[j] += 1.0
    return features

def extract_features(image):
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)
    bow = load_file_pickle(BOW_FILE_PICKLE)
    num_clusters = bow.shape[0]
    if descriptors is not None:
        return create_feature_bow(descriptors, bow, num_clusters)
    return np.zeros(num_clusters)

def predict(filepath):
    preprocessed_path, preprocessed_image = preprocess_image(filepath)
    features = extract_features(preprocessed_image)
    scaler = load_file_pickle(SCALER_FILE_PICKLE)
    scaled_features = scaler.transform([features])
    svm_model = load_file_pickle(SVM_FILE_PICKLE)
    probabilities = svm_model.predict_proba(scaled_features)[0]
    label = svm_model.classes_[np.argmax(probabilities)]
    confidence = round(np.max(probabilities) * 100, 2)
    return preprocessed_path, label, confidence
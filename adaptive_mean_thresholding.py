
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis, skew
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pywt
import joblib

def resize_and_convert_to_gray(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (512, 512))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(grayscale_image)
    return equalized_image

def adaptive_mean_thresholding(image):
    thresholded_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, blockSize=11, C=2)
    return thresholded_image

def kmeans_segmentation(image, num_clusters):
    flattened_image = image.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(flattened_image)
    cluster_labels = kmeans.labels_.reshape(image.shape)
    return cluster_labels

def segment_clusters(image, cluster_labels):
    segmented_image = np.zeros_like(image)
    segmented_image[cluster_labels == 0] = 0
    segmented_image[cluster_labels == 1] = 255
    return segmented_image

def extract_features(image):
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    idm = np.sum(glcm / (1 + (np.arange(256) - np.arange(256)) ** 2))
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    mean = np.mean(image)
    entropy_value = shannon_entropy(image)
    kurtosis_value = kurtosis(image.flatten())
    skewness_value = skew(image.flatten())
    coarseness = entropy(image, disk(5)).mean()
    directional_moment = np.sum(glcm**2)
    return [contrast, correlation, energy, mean, entropy_value, kurtosis_value,
            skewness_value, coarseness, directional_moment, idm]

input_folders = {'no': '/content/drive/My Drive/Detection/no',
                 'yes': '/content/drive/My Drive/Detection/Yes'}

all_features = []
all_labels = []

for label, folder_path in input_folders.items():
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(folder_path, filename)
            grayscale_image = resize_and_convert_to_gray(image_path)
            thresholded_image = adaptive_mean_thresholding(grayscale_image)
            cluster_labels = kmeans_segmentation(thresholded_image, 2)
            segmented_image = segment_clusters(thresholded_image, cluster_labels)
            features = extract_features(segmented_image)
            all_features.append(features)
            all_labels.append(label)

all_features_array = np.array(all_features)
all_labels_array = np.array(all_labels)

X_train, X_test, y_train, y_test = train_test_split(all_features_array, all_labels_array, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_imputed, y_train)

y_pred = svm_classifier.predict(X_test_imputed)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and imputer for future use
joblib.dump(svm_classifier, '/content/svm_classifier.pkl')
joblib.dump(imputer, '/content/imputer.pkl')

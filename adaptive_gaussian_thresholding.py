import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis, skew
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pywt
import warnings

warnings.filterwarnings("ignore")

def resize_and_convert_to_gray(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to load image at {image_path}.")
        return None
    resized_image = cv2.resize(image, (512, 512))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def adaptive_gaussian_thresholding(image):
    thresholded_image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2
    )
    return thresholded_image

def kmeans_segmentation(image, num_clusters=2):
    flattened_image = image.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(flattened_image)
    cluster_labels = kmeans.labels_.reshape(image.shape)
    return cluster_labels

def segment_clusters(image, cluster_labels):
    segmented_image = np.zeros_like(image)
    if np.mean(image[cluster_labels == 0]) > np.mean(image[cluster_labels == 1]):
        segmented_image[cluster_labels == 0] = 255
    else:
        segmented_image[cluster_labels == 1] = 255
    return segmented_image

def extract_features(image):
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs

    glcm = graycomatrix(
        image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256, symmetric=True, normed=True
    )

    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    mean = np.mean(cA)
    std_dev = np.std(cA)
    entropy_val = shannon_entropy(cA)
    kurtosis_val = kurtosis(cA.flatten())
    skewness_val = skew(cA.flatten())

    features = [
        contrast, correlation, energy, homogeneity,
        mean, std_dev, entropy_val, kurtosis_val, skewness_val
    ]
    return features

input_folders = {
    'no': r'/content/drive/My Drive/Detection/no',
    'yes': r'/content/drive/My Drive/Detection/Yes'
}

all_features = []
all_labels = []

for label, folder_path in input_folders.items():
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            grayscale_image = resize_and_convert_to_gray(image_path)

            if grayscale_image is None:
                continue

            thresholded_image = adaptive_gaussian_thresholding(grayscale_image)
            cluster_labels = kmeans_segmentation(thresholded_image)
            segmented_image = segment_clusters(grayscale_image, cluster_labels)
            features = extract_features(segmented_image)

            features = np.nan_to_num(features)

            all_features.append(features)
            all_labels.append(label)

X = np.array(all_features)
y = np.array(all_labels)

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
svm_classifier.fit(X_train, y_train)


y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

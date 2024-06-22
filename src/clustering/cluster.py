import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import glob

def extract_features(image_path, model, target_size=(224, 224)):
    """Extract features from an image using a pre-trained model."""
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

def main():
    print("Clustering images")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # loop over all images in cover1 folder and plt show them
    features = [extract_features(image, model) for image in glob.glob("cover1/*.jpg")]
    features2 = [extract_features(image, model) for image in glob.glob("cover2/*.jpg")]
    features.extend(features2)
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    kmeans.feature_names_in_ = ['cover1', 'cover2']
    print(kmeans.labels_)
    # print how many in each label
    print(np.sum(kmeans.labels_ == 0))
    print(np.sum(kmeans.labels_ == 1))

if __name__ == "__main__":
    main() 
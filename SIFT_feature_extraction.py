# https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
import pathlib

import cv2
import numpy as np
import scipy
from scipy.misc import imread
import pickle
import random
import os
import matplotlib.pyplot as plt


def extract_features(image_path, vector_size=32):
    image = imread(image_path, mode="RGB")
    try:
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints = sift.detect(image, None)

        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:vector_size]

        # computing descriptors vector
        keypoints, descriptors = sift.compute(image, keypoints)

        # Flatten all of them in one big vector - our feature vector
        descriptors = descriptors.flatten()
        # dsc = [x for xs in dsc for x in xs]

        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if len(descriptors) < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            descriptors = np.concatenate([descriptors, np.zeros(needed_size - len(descriptors))])
    except cv2.error as e:
        print('Error: ', e)
        return None

    return descriptors


def pickle_features(keypoints, descriptors):
    pickle_dsc = []
    for keypoint, descriptor in zip(keypoints, descriptors):
        pickle_dsc.append((keypoint.pt, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id, descriptor))
    return pickle_dsc


def batch_extractor(images_path, pickled_db_path="features.pck"):
    # files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    files = []

    for path, subdirs, dir_files in os.walk(images_path):
        for name in dir_files:
            files.append(os.path.join(path, name))

    result = {}
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)

    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)


class Matcher(object):

    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path, 'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=5):
        features = extract_features(image_path)
        img_distances = self.cos_cdist(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()


def show_img(path):
    img = imread(path, mode="RGB")
    plt.imshow(img)
    plt.show()


def train():
    images_path = 'train/'
    batch_extractor(images_path)

def evaluate():
    images_path = 'train/'
    files = []

    for path, subdirs, dir_files in os.walk(images_path):
        for name in dir_files:
            files.append(os.path.join(path, name))

    # files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    sample = random.sample(files, 3)

# this will be removed when I split train and evaluate actually
    batch_extractor(images_path)

    ma = Matcher('features.pck')

    for s in sample:
        print('Query image ==========================================')
        show_img(s)
        names, match = ma.match(s, topn=3)
        print('Result images ========================================')
        for i in range(3):
            print('Match %s' % (1 - match[i]))
            show_img(os.path.join(images_path, names[i]))


evaluate()
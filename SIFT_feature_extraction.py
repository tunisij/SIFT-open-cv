# https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774

import cv2
import imageio
import numpy as np
import scipy.spatial
import pickle
import os
import matplotlib.pyplot as plt


def extract_features(image_path, vector_size=32):
    image = imageio.imread(image_path)
    try:
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints = sift.detect(image, None)
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:vector_size]

        keypoints, descriptors = sift.compute(image, keypoints)

        descriptors = descriptors.flatten()

        needed_size = (vector_size * 64)
        if len(descriptors) < needed_size:
            descriptors = np.concatenate([descriptors, np.zeros(needed_size - len(descriptors))])
    except cv2.error as e:
        print('Error: ', e)
        return None

    return descriptors


def pickle_features(keypoints, descriptors):
    pickle_dsc = []
    for keypoint, descriptor in zip(keypoints, descriptors):
        pickle_dsc.append((keypoint.pt, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave,
                           keypoint.class_id, descriptor))
    return pickle_dsc


def batch_extractor(images_path, pickled_db_path="features.pck"):
    files = []
    for path, subdirs, dir_files in os.walk(images_path):
        for name in dir_files:
            files.append(os.path.join(path, name))

    result = {}
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)

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
    img = imageio.imread(path)
    plt.imshow(img)
    plt.show()


def train():
    images_path = 'train/'
    batch_extractor(images_path)

def evaluate():
    images_path = 'evaluate/'
    sample = []
    for path, subdirs, dir_files in os.walk(images_path):
        for name in dir_files:
            sample.append(os.path.join(path, name))

    ma = Matcher('features.pck')

    for s in sample:
        print('Evaluating top 3 matches for image:', s)
        names, match = ma.match(s, topn=5)
        for i in range(5):
            print('Matches %s\t%s' % (names[i], 1 - match[i]))
        print('\n')

# train()
evaluate()

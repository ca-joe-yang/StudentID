#! python3

import os
import sys
import face_recognition
import numpy as np

import scipy.misc
from sklearn.metrics.pairwise import euclidean_distances

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_gallery(dirname):
    im_names = os.listdir(dirname)
    im_names = [ im_name for im_name in im_names if im_name.endswith('.jpg') ]
    im_names.sort()
    embeds = []
    labels = []
    for im_name in im_names:
        print('Processing {}'.format(im_name))
        fname = os.path.join(dirname, im_name)
        image = face_recognition.load_image_file(fname)
        embed = face_recognition.face_encodings(image)[0]
        label = im_name.split('.')[0]
        embeds.append(embed)
        labels.append(label)

    gallery = dict(
        embeds=embeds,
        labels=labels
    )
    return gallery

def process_gallery(dirname):
    im_names = os.listdir(dirname)
    im_names.sort()
    for im_name in im_names:
        print('Processing {}'.format(im_name))
        fname = os.path.join(dirname, im_name)
        raw_image = face_recognition.load_image_file(fname)
        faces = get_faces(raw_image)
        if len(faces) == 0:
            raise ValueError('No faces found in {}'.format(fname))
        elif len(faces) > 1:
            raise ValueError('More that one faces in {}'.format(fname))
        student_id = im_name.split('.')[0]
        scipy.misc.imsave('gallery/{}.jpg'.format(student_id), faces[0])
        
def get_faces(image, use_gpu=True):
    if use_gpu:
        face_locations = face_recognition.face_locations(image, model='cnn')
    else:
        face_locations = face_recognition.face_locations(image, model='hog')
    faces = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face = image[top:bottom, left:right]
        faces.append(face)
    return faces, face_locations

if __name__ == '__main__':
    input_filename = sys.argv[1]
    input_image = face_recognition.load_image_file(input_filename)
    input_faces, input_face_locations = get_faces(input_image)

    input_embeds = face_recognition.face_encodings(input_image, input_face_locations)
    for i, face in enumerate(input_faces):
        scipy.misc.imsave('face_{}.jpg'.format(i), face)

    #process_gallery('raw')
    gallery = get_gallery('raw')

    distances = euclidean_distances(input_embeds, gallery['embeds'])
    #distances = distances / np.max(distances, axis=1).reshape(-1,1)
    sim = np.exp(-distances)
    gamma = 10
    score = [ softmax(gamma*s) for s in sim ]

    for i, face in enumerate(input_faces):
        scipy.misc.imsave('face_{}.jpg'.format(i), face)
        pred = np.argmax(score[i])
        print('Guess face_{} is student {} with confidence score {:.3f}'.format(
                i,
                gallery['labels'][pred],
                score[i][pred],
            ))




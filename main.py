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

def load_gallery(dirname):
    im_names = os.listdir(dirname)
    im_names = [ im_name for im_name in im_names if im_name.endswith('.jpg') ]
    im_names.sort()
    embeds = []
    labels = []
    for im_name in im_names:
        print('Processing {}'.format(im_name))
        fname = os.path.join(dirname, im_name)
        image = scipy.misc.imread(fname, mode='RGB')
        face_locations = [[0, image.shape[1], image.shape[0], 0]]
        embed = face_recognition.face_encodings(image, face_locations)[0]
        label = im_name.split('.')[0]
        embeds.append(embed)
        labels.append(label)

    gallery = dict(
        embeds=embeds,
        labels=labels
    )
    return gallery

def get_center_xy(location):
    top, right, bottom, left = location
    c_x = int((right+left)/2)
    c_y = int((top+bottom)/2)
    return c_x, c_y

def l2_dist(p1, p2):
    diff_x = p1[0] - p2[0]
    diff_y = p1[1] - p2[1]
    return np.sqrt(diff_x**2  + diff_y**2)

def raw2gallery(input_dirname='raw', output_dirname='gallery', use_gpu=False):
    im_names = os.listdir(input_dirname)
    im_names.sort()

    if not os.path.exists(output_dirname):
        os.mkdir(output_dirname)
    for im_name in im_names:
        print('Processing {}'.format(im_name))
        fname = os.path.join(input_dirname, im_name)
        raw_image = scipy.misc.imread(fname, mode='RGB')
        faces, _ = get_faces(raw_image, use_gpu=use_gpu)
        if len(faces) == 0:
            raise ValueError('No faces found in {}'.format(fname))
        elif len(faces) > 1:
            raise ValueError('More that one faces found in {}'.format(fname))
        student_id = im_name.split('.')[0]

        fname = os.path.join(output_dirname, '{}.jpg'.format(student_id))
        scipy.misc.imsave(fname, faces[0])
        
def get_faces(image, use_gpu=False):
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

def seat_configuration(input_filename, output_filename='seat.csv', seat_row=3, seat_column=4):
    seat_image = scipy.misc.imread(input_filename, mode='RGB')
    _, seat_locations = get_faces(seat_image)

    seat_xy = [get_center_xy(i) for i in seat_locations]
    seat_xy = sorted(seat_xy, key=lambda x: x[1])
    
    seat_xy_sorted = []
    for i in range(seat_row):
        tmp = seat_xy[i*seat_column:i*seat_column+seat_column]
        tmp_sorted = sorted(tmp, key=lambda x: x[0])
        for j, (x,y) in enumerate(tmp_sorted):
            seat_xy_sorted.append([i,j,x,y])

    np.savetxt('seat.csv', seat_xy_sorted, fmt=['%d', '%d', '%d', '%d'], delimiter=',')

def load_seat_center(filename):
    seat = np.genfromtxt(filename, delimiter=',', dtype=int)
    seat_row = np.max(seat[:,0]) + 1
    seat_column = np.max(seat[:,1]) + 1
    seat_centers = np.zeros([seat_row, seat_column, 2])
    for r, c, x, y in seat:
        seat_centers[r,c] = [x,y]
    return seat_centers

if __name__ == '__main__':
    debug = False
    use_gpu = False
    raw2gallery(use_gpu=use_gpu)
    #seat_configuration('seat.png')
    gallery = load_gallery('gallery')

    query_filename = sys.argv[1]
    query_image = scipy.misc.imread(query_filename, mode='RGB')
    query_faces, query_face_locations = get_faces(query_image, use_gpu=use_gpu)

    query_embeds = face_recognition.face_encodings(query_image, query_face_locations)
    if debug:
        for i, face in enumerate(query_faces):
            scipy.misc.imsave('face_{}.jpg'.format(i), face)

    distances = euclidean_distances(query_embeds, gallery['embeds'])
    #distances = distances / np.max(distances, axis=1).reshape(-1,1)
    sim = np.exp(-distances)
    gamma = 10
    score = [ softmax(gamma*s) for s in sim ]

    seat_centers = load_seat_center('seat.csv')
    seat_row, seat_column = seat_centers.shape[:2]

    student_seats = np.chararray([seat_row, seat_column], itemsize=15)
    for i, (face, face_location) in enumerate(zip(query_faces, query_face_locations)):
        face_center = get_center_xy(face_location)

        D = np.array(seat_centers) - np.array(face_center)
        B = np.sum(np.square(D), axis=2)
        seat_idx = np.unravel_index(np.argmin(B), B.shape)
        pred = np.argmax(score[i])

        print('Guess face_{} is student {} at seat-{} with confidence score {:.3f}'.format(
                i,
                gallery['labels'][pred],
                seat_idx,
                score[i][pred],
            ))

        if score[i][pred] < 0.35:
            student_seats[seat_idx] = 'Unknown'
        else:
            student_seats[seat_idx] = gallery['labels'][pred]

    for r in range(seat_row):
        print('#'*77)
        line = ['']
        for c in range(seat_column):
            line.append('   {:15s}'.format(student_seats[r,c].decode("utf-8") ))
        line.append('')
        line = '#'.join(line)
        print(line)
    print('#'*77)

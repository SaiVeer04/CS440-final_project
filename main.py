import numpy as np
import os
from classifiers import Perceptron

def load_face_data(data_path, labels_path):
    with open(data_path, 'r') as f:
        content = f.readlines()
 
    faces = []
    current_face = []
    for line in content:
        line = line.rstrip('\n')
        if len(current_face) < 70:
            current_face.append(line)
        else:
            faces.append('\n'.join(current_face))
            current_face = [line]
    
    if current_face:
        faces.append('\n'.join(current_face))
    
    with open(labels_path, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    
    
    face_vectors = []
    for face in faces:
        vector = []
        for c in face:
            if c == "#":
                vector.append(1)
            elif c == ' ':
                vector.append(0)
        
        face_vectors.append(vector)
    
    return np.array(face_vectors, dtype=np.float32), np.array(labels, dtype=np.int64)

def load_digit_data(data_path, labels_path):
    with open(data_path, 'r') as f:
        content = f.readlines()
    
    digits = []
    current_digits = []
    for line in content:
        line = line.rstrip('\n')
        if len(current_digits) < 28:
            current_digits.append(line)
        else:
            digits.append('\n'.join(current_digits))
            current_digits = [line]

    
    if current_digits:
        digits.append('\n'.join(current_digits))
    
    with open(labels_path, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    
    digit_vectors = []
    for digit in digits:
        vector = []
        for c in digit:
            if c == "#":
                vector.append(1)
            elif c == ' ' or c == '+':
                vector.append(0)
        
        digit_vectors.append(vector)
    
    return np.array(digit_vectors, dtype=np.float32), np.array(labels, dtype=np.int64)

x_values, y_values = load_face_data("data/facedata/facedatatrain" , "data/facedata/facedatatrainlabels")

digit_data, digit_labels = load_digit_data("data/digitdata/trainingimages","data/digitdata/traininglabels")

model = Perceptron(input_size=x_values.shape[1], num_classes=2)
model.train(x_values, y_values)

x_test_values, y_test_values = load_face_data("data/facedata/facedatatest" , "data/facedata/facedatatestlabels")

print(model.evaluate(x_test_values, y_test_values))


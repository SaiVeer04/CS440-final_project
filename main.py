import numpy as np
import os
from classifiers import Perceptron
from classifiers import CustomNN
from pytorch_nn import TorchNN, TrainerConfig, train_and_evaluate
import torch
import time


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


def main():
    # 1. Use perceptron to evaluate digit data
    x_values, y_values = load_face_data("data/facedata/facedatatrain", "data/facedata/facedatatrainlabels")

    digit_data, digit_labels = load_digit_data("data/digitdata/trainingimages", "data/digitdata/traininglabels")

    model = Perceptron(input_size=digit_data.shape[1], num_classes=10)
    model.train(digit_data, digit_labels)

    x_test_values, y_test_values = load_face_data("data/facedata/facedatatest", "data/facedata/facedatatestlabels")
    digit_test_data, digit_test_labels = load_digit_data("data/digitdata/testimages", "data/digitdata/testlabels")

    print(f"Evaluation on Perceptron digit data: {model.evaluate(digit_test_data, digit_test_labels)}")

    # 2. Use CustomNN constructed to evaluate accuracy of classifying face data
    model2 = CustomNN(input_size=x_values.shape[1], hidden_size1=256, hidden_size2=128, output_size=2)
    model2.train(x_values, y_values)

    print(f"Evaluation on CustomNN face data: {model2.evaluate(x_test_values, y_test_values)}")

    # 3. PyTorch based experiment
    X_train_np, y_train_np = load_digit_data(
        "data/digitdata/trainingimages",
        "data/digitdata/traininglabels"
    )
    X_test_np, y_test_np = load_digit_data(
        "data/digitdata/testimages",
        "data/digitdata/testlabels"
    )

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    model = TorchNN(
        input_size=X_train.shape[1],
        hidden1=256,
        hidden2=128,
        output_size=10
    )
    cfg = TrainerConfig(
        lr=0.01,
        batch_size=128,
        epochs=10
    )

    start = time.time()
    metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test, cfg)
    duration = time.time() - start

    print(
        f"Evaluation on PyTorchNN — Test Acc: {metrics['val_accuracy']:.4f}, "
        f"Train Acc: {metrics['train_accuracy']:.4f}, "
        f"Time: {duration:.2f}s"
    )


if __name__ == '__main__':
    main()

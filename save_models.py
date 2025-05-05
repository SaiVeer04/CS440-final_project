from main import load_digit_data, load_face_data
from classifiers import Perceptron
from classifiers import CustomNN
from pytorch_nn import TorchNN, TrainerConfig, train_model
import torch
import numpy as np

face_train_data, face_train_labels = load_face_data("data/facedata/facedatatrain", "data/facedata/facedatatrainlabels")


digit_train_data, digit_train_labels = load_digit_data("data/digitdata/trainingimages", "data/digitdata/traininglabels")


def save_perceptron_models():
    face_model_perceptron = Perceptron(input_size=face_train_data.shape[1], num_classes= 2)
    face_model_perceptron.train(face_train_data, face_train_labels)
    np.savez("models/face_perceptron_weights.npz", weights = face_model_perceptron.weights, bias = face_model_perceptron.bias)

    digit_model_perceptron = Perceptron(input_size=digit_train_data.shape[1], num_classes=10)
    digit_model_perceptron.train(digit_train_data, digit_train_labels)
    np.savez("models/digit_perceptron_weights.npz", weights = digit_model_perceptron.weights, bias = digit_model_perceptron.bias)

def save_custom_nn_models():
    face_model_customNN = CustomNN(input_size=face_train_data.shape[1], hidden_size1=256, hidden_size2=128, output_size=2)
    face_model_customNN.train(face_train_data, face_train_labels)
    np.savez("models/face_customNN_weights.npz",
         W1=face_model_customNN.W1, b1=face_model_customNN.b1,
         W2=face_model_customNN.W2, b2=face_model_customNN.b2,
         W3=face_model_customNN.W3, b3=face_model_customNN.b3)
    
    digit_model_customNN = CustomNN(input_size=digit_train_data.shape[1], hidden_size1=256, hidden_size2=128, output_size=10)
    digit_model_customNN.train(digit_train_data, digit_train_labels)
    np.savez("models/digit_customNN_weights.npz",
         W1=digit_model_customNN.W1, b1=digit_model_customNN.b1,
         W2=digit_model_customNN.W2, b2=digit_model_customNN.b2,
         W3=digit_model_customNN.W3, b3=digit_model_customNN.b3)

def save_pytorch_models():
    face_X_train = torch.tensor(face_train_data, dtype=torch.float32)
    face_y_train = torch.tensor(face_train_labels, dtype=torch.long)
    face_model = TorchNN(
        input_size=face_X_train.shape[1],
        hidden1=256,
        hidden2=128,
        output_size=2
    )

    digit_X_train = torch.tensor(digit_train_data, dtype=torch.float32)
    digit_y_train = torch.tensor(digit_train_labels, dtype=torch.long)
    digit_model = TorchNN(
        input_size=digit_X_train.shape[1],
        hidden1=256,
        hidden2=128,
        output_size=10
    )

    cfg = TrainerConfig(
        lr=0.01,
        batch_size=128,
        epochs=10
    )

    face_model_pytorch = train_model(face_model, face_X_train, face_y_train, cfg)
    digit_model_pytorch = train_model(digit_model, digit_X_train, digit_y_train, cfg)

    torch.save(face_model_pytorch.state_dict(), "models/face_pytorch_weights.pth")
    torch.save(digit_model_pytorch.state_dict(), "models/digit_pytorch_weights.pth")

def load_perceptron_models():
    face_data = np.load("models/face_perceptron_weights.npz")
    face_model = Perceptron(input_size=face_data['weights'].shape[0], num_classes= 2)

    digit_data = np.load("models/digit_perceptron_weights.npz")
    digit_model = Perceptron(input_size=digit_data['weights'].shape[0], num_classes= 10)

    face_model.weights = face_data['weights']
    face_model.bias = face_data['bias']

    digit_model.weights = digit_data['weights']
    digit_model.bias = digit_data['bias']

    return face_model, digit_model

def load_custom_nn_models():
    face_data = np.load("models/face_customnn_weights.npz")
    face_model = CustomNN(
        input_size=face_data['W1'].shape[0], 
        hidden_size1=face_data['W1'].shape[1], 
        hidden_size2=face_data['W2'].shape[1], 
        output_size=2)
    
    digit_data = np.load("models/digit_customnn_weights.npz")
    digit_model = CustomNN(
        input_size=digit_data['W1'].shape[0], 
        hidden_size1=digit_data['W1'].shape[1], 
        hidden_size2=digit_data['W2'].shape[1], 
        output_size=2)
    
    face_model.W1 = face_data['W1']
    face_model.b1 = face_data['b1']
    face_model.W2 = face_data['W2']
    face_model.b2 = face_data['b2']
    face_model.W3 = face_data['W3']
    face_model.b3 = face_data['b3']

    digit_model.W1 = digit_data['W1']
    digit_model.b1 = digit_data['b1']
    digit_model.W2 = digit_data['W2']
    digit_model.b2 = digit_data['b2']
    digit_model.W3 = digit_data['W3']
    digit_model.b3 = digit_data['b3']

    return face_model, digit_model

def load_pytorch_models():
    face_X_train = torch.tensor(face_train_data, dtype=torch.float32)
    digit_X_train = torch.tensor(digit_train_data, dtype=torch.float32)
    
    face_model = TorchNN(
        input_size=face_X_train.shape[1],
        hidden1=256,
        hidden2=128,
        output_size=2
    )
    face_model.load_state_dict(torch.load("models/face_pytorch_weights.pth"))

    digit_model = TorchNN(
        input_size=digit_X_train.shape[1],
        hidden1=256,
        hidden2=128,
        output_size=10
    )
    digit_model.load_state_dict(torch.load("models/digit_pytorch_weights.pth"))

    return face_model, digit_model







#save_perceptron_models()
#save_custom_nn_models()
#save_pytorch_models()
    










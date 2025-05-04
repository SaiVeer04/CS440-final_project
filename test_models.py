from save_models import load_custom_nn_models, load_perceptron_models, load_pytorch_models
from main import load_digit_data, load_face_data
from classifiers import Perceptron
from classifiers import CustomNN
from pytorch_nn import TorchNN, make_dataloader
import torch
import numpy as np
import matplotlib.pyplot as plt


face_perceptron, digit_perceptron = load_perceptron_models()
face_custom_nn, digit_custom_nn = load_custom_nn_models()
face_pytorch, digit_pytorch = load_pytorch_models()

face_test_data, face_test_labels  = load_face_data("data/facedata/facedatatest", "data/facedata/facedatatestlabels")
digit_test_data, digit_test_labels = load_digit_data("data/digitdata/testimages", "data/digitdata/testlabels")

def get_face(data_path, idx):
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
    
    return faces[idx]

def get_digit(data_path, idx):
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
    
    return digits[idx]

    

#1a. Test perceptron (face)
idx = np.random.randint(0, face_test_data.shape[0])
x_single = face_test_data[idx:idx+1] 
y_true = face_test_labels[idx]

y_pred = face_perceptron.predict(x_single)[0]
print(f"Perceptron Face Classification")
print(f"True Label: {y_true}, Predicted Label: {y_pred}")
print()

face = get_face("data/facedata/facedatatest", idx)

fig, ax = plt.subplots()
font = {'family': 'monospace',
        'size': 3}
ax.text(0.05, 0.95, face, transform=ax.transAxes, fontsize=font['size'], fontfamily=font['family'], verticalalignment='top')
ax.axis('off')
plt.title(f"Perceptron Face Classification: (True Label: {y_true}, Predicted Label: {y_pred})")
plt.show()

#1b. Test Perceptron (digits)
idx = np.random.randint(0, digit_test_data.shape[0])
x_single = digit_test_data[idx:idx+1] 
y_true = digit_test_labels[idx]

y_pred = digit_perceptron.predict(x_single)[0]
print(f"Perceptron Digit Classification")
print(f"True Label: {y_true}, Predicted Label: {y_pred}")
print()

digit = get_digit("data/digitdata/testimages", idx)

fig, ax = plt.subplots()
font = {'family': 'monospace',
        'size': 3}
ax.text(0.05, 0.95, digit, transform=ax.transAxes, fontsize=font['size'], fontfamily=font['family'], verticalalignment='top')
ax.axis('off')
plt.title(f"Perceptron Digit Classification: (True Label: {y_true}, Predicted Label: {y_pred})")
plt.show()

#2a. Test CustomNN (face)
idx = np.random.randint(0, face_test_data.shape[0])
x_single = face_test_data[idx:idx+1] 
y_true = face_test_labels[idx]

y_pred = face_custom_nn.predict(x_single)[0]
print(f"CustomNN Face Classification")
print(f"True Label: {y_true}, Predicted Label: {y_pred}")
print()

face = get_face("data/facedata/facedatatest", idx)

fig, ax = plt.subplots()
font = {'family': 'monospace',
        'size': 3}
ax.text(0.05, 0.95, face, transform=ax.transAxes, fontsize=font['size'], fontfamily=font['family'], verticalalignment='top')
ax.axis('off')
plt.title(f"CustomNN Face Classification: (True Label: {y_true}, Predicted Label: {y_pred})")
plt.show()

#2b. Test CustomNN (digits)
idx = np.random.randint(0, digit_test_data.shape[0])
x_single = digit_test_data[idx:idx+1] 
y_true = digit_test_labels[idx]

y_pred = digit_custom_nn.predict(x_single)[0]
print(f"CustomNN Digit Classification")
print(f"True Label: {y_true}, Predicted Label: {y_pred}")
print()

digit = get_digit("data/digitdata/testimages", idx)

fig, ax = plt.subplots()
font = {'family': 'monospace',
        'size': 3}
ax.text(0.05, 0.95, digit, transform=ax.transAxes, fontsize=font['size'], fontfamily=font['family'], verticalalignment='top')
ax.axis('off')
plt.title(f"CustomNN Digit Classification: (True Label: {y_true}, Predicted Label: {y_pred})")
plt.show()

# data loader for pytorch
face_X_test = torch.tensor(face_test_data, dtype=torch.float32)
face_y_test = torch.tensor(face_test_labels, dtype=torch.long)
digit_X_test = torch.tensor(digit_test_data, dtype=torch.float32)
digit_y_test = torch.tensor(digit_test_labels, dtype=torch.long)


#3a. Test PytorchNN (face)
idx = torch.randint(len(face_X_test), (1,)).item()
x_single = face_X_test[idx].unsqueeze(0)
y_true = face_y_test[idx].item()

if torch.cuda.is_available():
    device = torch.device("cuda") 
    
else:
    device = torch.device("cpu")

x_single = x_single.to(device)
face_pytorch.eval()

with torch.no_grad():
    logits = face_pytorch(x_single)
    y_pred = logits.argmax(dim=1).item()
print("PyTorch NN Face Classification")
print(f"True Label: {y_true}, Predicted Label: {y_pred}")
print()

face = get_face("data/facedata/facedatatest", idx)
fig, ax = plt.subplots()
font = {'family': 'monospace',
        'size': 3}
ax.text(0.05, 0.95, face, transform=ax.transAxes, fontsize=font['size'], fontfamily=font['family'], verticalalignment='top')
ax.axis('off')
plt.title(f"PyTorchNN Face Classification: (True Label: {y_true}, Predicted Label: {y_pred})")
plt.show()

#3b. Test PyTorch Digit Classification
idx = torch.randint(len(digit_X_test), (1,)).item()
x_single = digit_X_test[idx].unsqueeze(0)
y_true = digit_y_test[idx].item()

x_single = x_single.to(device)
digit_pytorch.eval()

with torch.no_grad():
    logits = digit_pytorch(x_single)
    y_pred = logits.argmax(dim=1).item()

print("PyTorch NN Digit Classification")
print(f"True Label: {y_true}, Predicted Label: {y_pred}")

digit = get_digit("data/digitdata/testimages", idx)

fig, ax = plt.subplots()
font = {'family': 'monospace',
        'size': 3}
ax.text(0.05, 0.95, digit, transform=ax.transAxes, fontsize=font['size'], fontfamily=font['family'], verticalalignment='top')
ax.axis('off')
plt.title(f"PyTorchNN Digit Classification: (True Label: {y_true}, Predicted Label: {y_pred})")
plt.show()
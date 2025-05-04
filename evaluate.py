import csv
import time
import numpy as np
import torch
from main import load_face_data, load_digit_data
from classifiers import Perceptron, CustomNN
from pytorch_nn import TorchNN, TrainerConfig, train_and_evaluate
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

frameworks = {
    'scratch': ['perceptron', 'customNN'],
    'pytorch': ['customNN']
}
datasets = ['digits', 'faces']
percents = list(range(10, 101, 10))
trials = 5


def sample_data(X, y, percent, seed=None):
    if seed is not None:
        np.random.seed(seed)
    idx = np.random.permutation(len(y))
    n = max(1, int(len(y) * percent / 100))
    sel = idx[:n]
    return X[sel], y[sel]


with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['framework', 'model', 'dataset', 'percent', 'acc_mean', 'acc_std', 'time_mean', 'time_std'])
    for fw, models in frameworks.items():
        for model in models:
            for ds in datasets:
                for pct in percents:
                    accs = []
                    times = []
                    for t in range(trials):
                        if ds == 'digits':
                            X_train, y_train = load_digit_data(
                                "data/digitdata/trainingimages",
                                "data/digitdata/traininglabels"
                            )
                            X_test, y_test = load_digit_data(
                                "data/digitdata/testimages",
                                "data/digitdata/testlabels"
                            )
                            num_classes = 10
                        else:
                            # faces
                            X_train, y_train = load_face_data(
                                "data/facedata/facedatatrain",
                                "data/facedata/facedatatrainlabels"
                            )
                            X_test, y_test = load_face_data(
                                "data/facedata/facedatatest",
                                "data/facedata/facedatatestlabels"
                            )
                            num_classes = 2
                        Xs, ys = sample_data(X_train, y_train, pct, seed=t)
                        start = time.time()
                        if fw == 'scratch':
                            if model == 'perceptron':
                                clf = Perceptron(input_size=Xs.shape[1], num_classes=num_classes)
                                perceptron_start = time.time()
                                clf.train(Xs, ys)
                                duration = time.time() - perceptron_start
                                acc = clf.evaluate(X_test, y_test)
                                logger.info(f"[scratch/perceptron] pct={pct:>3}%  trial={t}  "
                                            f"acc={acc:.4f}  time={duration:.2f}s")
                            elif model == 'customNN':
                                # custom nn
                                clf = CustomNN(input_size=Xs.shape[1],
                                               hidden_size1=256,
                                               hidden_size2=128,
                                               output_size=num_classes)
                                custom_nn_start = time.time()
                                clf.train(Xs, ys)
                                duration = time.time() - custom_nn_start
                                acc = clf.evaluate(X_test, y_test)
                                logger.info(f"[scratch/customNN]   pct={pct:>3}%  trial={t}  acc={acc:.4f}  "
                                            f"time={duration:.2f}s")
                        else:
                            # pytorch
                            Xtr = torch.tensor(Xs, dtype=torch.float32)
                            ytr = torch.tensor(ys, dtype=torch.long)
                            Xte = torch.tensor(X_test, dtype=torch.float32)
                            yte = torch.tensor(y_test, dtype=torch.long)
                            clf = TorchNN(input_size=Xtr.shape[1],
                                          hidden1=256,
                                          hidden2=128,
                                          output_size=num_classes)
                            cfg = TrainerConfig(lr=0.01, batch_size=128, epochs=50)
                            metrics = train_and_evaluate(clf, Xtr, ytr, Xte, yte, cfg)
                            acc = metrics['val_accuracy']
                        duration = time.time() - start
                        accs.append(acc)
                        times.append(duration)
                    writer.writerow([
                        fw, model, ds, pct,
                        np.mean(accs), np.std(accs),
                        np.mean(times), np.std(times)
                    ])
print("Evaluation complete. Saved to results.csv")

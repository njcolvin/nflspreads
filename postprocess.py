import torch
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from constants import device
import numpy as np

def predict(model, X):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sample in X:
            sample = torch.from_numpy(sample).to(device)
            pred = model(sample)
            _, max_index = torch.max(pred, 0)
            predictions.append(int(max_index.cpu()))
    return predictions

def evaluate_models(my_model, Xtrn, Xtst, Ytrn, Ytst):
    mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(64, 128, 32), batch_size=64, activation='relu').fit(Xtrn, Ytrn)
    dt = DecisionTreeClassifier().fit(Xtrn, Ytrn)
    lr = LogisticRegression().fit(Xtrn, Ytrn)
    gd = GradientBoostingClassifier().fit(Xtrn, Ytrn)

    # evaluate models
    models = [mlp, dt, lr, gd]
    preds = [model.predict(Xtst) for model in models]
    models.append(my_model)
    preds.append(predict(my_model, Xtst))
    evals = [(recall_score(Ytst, pred, average='weighted'),
              precision_score(Ytst, pred, average='weighted'),
              f1_score(Ytst, pred, average='weighted')) for pred in preds]

    # print results
    for i in range(len(models)):
        print()
        print(type(models[i]).__name__)
        print(' Recall: {}'.format(evals[i][0]))
        print(' Precision: {}'.format(evals[i][1]))
        print(' F1 score: {}'.format(evals[i][2]))
        print(' actual: ' + str(Ytst[:5]))
        print(' predicted: ' + str(preds[i][:5]))
        unique, counts = np.unique(preds[i], return_counts=True)
        print(' ' + str(unique))
        print(' ' + str(counts))
    unique, counts = np.unique(Ytst, return_counts=True)
    print(unique)
    print(counts)
    
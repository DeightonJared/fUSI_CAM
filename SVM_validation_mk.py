import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from normalize import normalize_fUSI_bunch
from dataset import Mouse_Dataset
import random

np.random.seed(13)
k = 10
step_size = 60

mk1_5 = np.moveaxis(np.load('MK801_Exp/mk1_5_Afterregis.npy'), 3, 1)
sal = np.moveaxis(np.load('MK801_Exp/Sal_Afterregis.npy'), 3, 1)

print(mk1_5.shape, sal.shape)

mk_dat = normalize_fUSI_bunch(mk1_5)
sal_dat = normalize_fUSI_bunch(sal)

metrics = {
    'accuracy': {},
    'precision': {},
    'recall': {},
    'f1_score': {},
    'roc_auc_score': {},
    'confusion_matrix': {},
}

coefficients = {
    'MK': {},
    'Saline': {}
}

for step, t in enumerate(np.arange(0, 3600, step_size)):
    for key in metrics.keys():
        metrics[key][step] = []

mk_pop = range(mk_dat.shape[0])
sal_pop = range(sal_dat.shape[0])

training_options = []

for i in range(k):
    mk_indices = random.sample(mk_pop, 5)
    sal_indices = random.sample(sal_pop, 7)

    if (mk_indices not in [training_options[k][0] for k in range(len(training_options))]) and (sal_indices not in [training_options[k][1] for k in range(len(training_options))]):
        training_options.append((mk_indices, sal_indices))

for train_mk_indices, train_sal_indices in training_options:
    coefficients['MK'][str(train_mk_indices)] = []
    coefficients['Saline'][str(train_sal_indices)] = []

for train_mk_indices, train_sal_indices in training_options:
    mk_coeffs = []
    sal_coeffs = []
    for step, t in enumerate(np.arange(0, 3600, step_size)):
        
        train_dataset = Mouse_Dataset(mk_dat[train_mk_indices, t:t+step_size:, :, :], sal_dat[train_sal_indices, t:t+step_size:, :, :])

        mk_test_dat = np.delete(mk_dat, train_mk_indices, axis=0)
        sal_test_dat = np.delete(sal_dat, train_sal_indices, axis=0)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        print('Training SVM!')
        X_train, y_train = [], []
        for batch in train_dataloader:
            data, labels = batch
            X_train.append(data.view(data.size(0), -1).numpy())
            y_train.append(labels.numpy())
        
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        
        y_train = 2 * y_train - 1

        svm = LinearSVC(max_iter=10000)
        svm.fit(X_train, y_train)

        print(f'Finding scores in step {step + 1} of {3600 // step_size}')

        test_dataset = Mouse_Dataset(mk_test_dat[:, t:t+step_size, :, :], sal_test_dat[:, t:t+step_size, :, :])
        
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        
        X_test, y_true = [], []
        for batch in test_dataloader:
            data, labels = batch
            X_test.append(data.view(data.size(0), -1).numpy())
            y_true.append(labels.numpy())

        X_test = np.concatenate(X_test, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        
        y_true = 2 * y_true - 1
        
        y_pred = svm.predict(X_test)

        metrics['accuracy'][step].append(accuracy_score(y_true, y_pred))
        metrics['precision'][step].append(precision_score(y_true, y_pred, average='binary'))
        metrics['recall'][step].append(recall_score(y_true, y_pred, average='binary'))
        metrics['f1_score'][step].append(f1_score(y_true, y_pred, average='binary'))
        metrics['roc_auc_score'][step].append(roc_auc_score(y_true, y_pred))
        metrics['confusion_matrix'][step].append(confusion_matrix(y_true, y_pred))

        mk_coeffs.append(svm.coef_[0].reshape((91,128)))
        sal_coeffs.append(svm.coef_[0].reshape((91,128)))

    coefficients['MK'][str(train_mk_indices)].append(np.array(mk_coeffs))
    coefficients['Saline'][str(train_sal_indices)].append(np.array(sal_coeffs))

np.save('mk_results/mk1_5/validation_metrics_svm_pt3' + str(step_size), metrics)
np.save('mk_results/mk1_5/validation_coefficients_svm_pt3' + str(step_size), coefficients)

# Validation (Anasthesia)

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from normalize import normalize_fUSI_bunch
from dataset import Mouse_Dataset
from model import MyCNN
from trainer import train, test
import random
from torchcam.methods import XGradCAM
from viz import get_CAMs_anasthesia

torch.manual_seed(21)

k = 10
step_size = 60

sal = np.moveaxis(np.load('MK801_Exp/sal_Afterregis.npy'), 3, 1)
print(sal.shape)
sal_dat =  normalize_fUSI_bunch(sal)

metrics = {
    'accuracy': {},
    'precision': {},
    'recall': {},
    'f1_score': {},
    'roc_auc_score': {},
    'confusion_matrix': {}
}


CAMS = {'None':{},
        'Saline': {}
        }

for step, t in enumerate(np.arange(0, 3600, step_size)):
    for key in metrics.keys():
        metrics[key][step] = []


sal_pop = range(sal_dat.shape[0])

training_options = []

for i in range(k):
    none_indices = random.sample(sal_pop, 7) 
    sal_indices = random.sample(sal_pop, 7)
    
    if (none_indices not in [training_options[k][0] for k in range(len(training_options))]) and (sal_indices not in [training_options[k][1] for k in range(len(training_options))]):
        training_options.append((none_indices, sal_indices))

for train_none_indices, train_sal_indices in training_options:
    CAMS['None'][str(train_none_indices)] = []
    CAMS['Saline'][str(train_sal_indices)] = []
    
# Run Validation

for train_none_indices, train_sal_indices in training_options:
    # Create train/test datasets for each fold
    # train_dataset = Mouse_Dataset(sal_dat[train_none_indices, :300, :, :], sal_dat[train_sal_indices, 3300:, :, :])
    train_dataset = Mouse_Dataset(sal_dat[train_none_indices, 300:600, :, :], sal_dat[train_sal_indices, 3300:, :, :])
    
    none_test_dat = np.delete(sal_dat, train_none_indices, axis=0)
    sal_test_dat = np.delete(sal_dat, train_sal_indices, axis=0)
    
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
    
    # Model training and evaluation remains the same...
    model = MyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss()
    
    Xgrad = XGradCAM(model, target_layer = 'conv4')
    
    none_avg_CAMs = []
    sal_avg_CAMs = []
    none_avg_imgs = []
    sal_avg_imgs = []
    
    print('Training!')
    
    for i in range(5):
        train(train_dataloader, model, criterion, optimizer)
        

    for step, t in enumerate(np.arange(0, 3600, step_size)):
        
        print('finding scores in step ' +  str(step + 1) + ' of ' + str(3600/step_size))
        
        #test_dataset = Mouse_Dataset(none_test_dat[:, 240:300,:,:],
        #                             sal_test_dat[:,t:t+step_size,:,:])
        
        test_dataset = Mouse_Dataset(none_test_dat[:, 300:360,:,:], sal_test_dat[:,t:t+step_size,:,:])
        
        #none_num_images = none_test_dat[:, 240:300,:,:].shape[0]*none_test_dat[:, 240:300,:,:].shape[1]
        none_num_images = none_test_dat[:,300:360,:,:].shape[0]*none_test_dat[:, 300:360,:,:].shape[1]
        sal_num_images = sal_test_dat[:,t:t+step_size,:,:].shape[0]*sal_test_dat[:,t:t+step_size,:,:].shape[1]
        
        test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle=False)

        y_true, y_pred, _ = test(test_dataloader, model, criterion)
        
        metrics['accuracy'][step].append(accuracy_score(y_true, y_pred))
        metrics['precision'][step].append(precision_score(y_true, y_pred, average='binary'))
        metrics['recall'][step].append(recall_score(y_true, y_pred, average='binary'))
        metrics['f1_score'][step].append(f1_score(y_true, y_pred, average='binary'))
        metrics['roc_auc_score'][step].append(roc_auc_score(y_true, y_pred))
        metrics['confusion_matrix'][step].append(confusion_matrix(y_true, y_pred))
    
        none_avg_cam, sal_avg_cam, none_avg_img, sal_avg_img = get_CAMs_anasthesia(model, test_dataset, none_num_images = none_num_images, sal_num_images=sal_num_images, cam_extractor=Xgrad)
        
        none_avg_CAMs.append(none_avg_cam)
        sal_avg_CAMs.append(sal_avg_cam)
        none_avg_imgs.append(none_avg_img)
        sal_avg_imgs.append(sal_avg_imgs)
        
    NoneCAMS = torch.stack(none_avg_CAMs).numpy()
    salCAMS = torch.stack(sal_avg_CAMs).numpy()
    
    CAMS['None'][str(train_none_indices)].append(NoneCAMS)
    CAMS['Saline'][str(train_sal_indices)].append(salCAMS)
        


# Save the dictionary 
np.save('anasthesia_results/validation_metrics' + str(step_size), metrics)

np.save('anasthesia_results/validation_CAMS' + str(step_size), CAMS)


# Validation

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from normalize import normalize_fUSI_bunch
from dataset import Mouse_Dataset
from model import MyCNN_flexible
from trainer import train, test
import random
from viz import get_CAMs
from torchcam.methods import XGradCAM

torch.manual_seed(13)

k = 10
step_size = 60

mk1_5 = np.moveaxis(np.load('MK801_Exp/mk1_5_Afterregis.npy'), 3, 1)
# mk0_5 = np.moveaxis(np.load('MK801_Exp/mk0_5_Afterregis.npy'), 3, 1)

sal = np.moveaxis(np.load('MK801_Exp/Sal_Afterregis.npy'), 3, 1)

print(mk1_5.shape, sal.shape)

mk_dat = normalize_fUSI_bunch(mk1_5)
sal_dat =  normalize_fUSI_bunch(sal)

metrics = {
    'accuracy': {},
    'precision': {},
    'recall': {},
    'f1_score': {},
    'roc_auc_score': {},
    'confusion_matrix': {},
}

CAMS = {'MK':{},
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
    CAMS['MK'][str(train_mk_indices)] = []
    CAMS['Saline'][str(train_sal_indices)] = []

# Run Validation

for train_mk_indices, train_sal_indices in training_options:
    # Create train/test datasets for each fold
    train_dataset = Mouse_Dataset(mk_dat[train_mk_indices, 3300:, :, :], sal_dat[train_sal_indices, 3300:, :, :])
    
    mk_test_dat = np.delete(mk_dat, train_mk_indices, axis=0)
    sal_test_dat = np.delete(sal_dat, train_sal_indices, axis=0)
    
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
    
    #model = MyCNN_flexible()
    model = MyCNN_flexible(num_filters = 64, num_conv_layers=4, activation='ELU')
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)#1e-3)
    criterion = nn.CrossEntropyLoss()
    
    Xgrad = XGradCAM(model, target_layer=list(model.named_modules())[-2][0])
    
    print('Training!')
    for i in range(5):#range(3):
        train(train_dataloader, model, criterion, optimizer)
        
    
    mk_avg_CAMs = []
    sal_avg_CAMs = []
    mk_avg_imgs = []
    sal_avg_imgs = []
    
    for step, t in enumerate(np.arange(0, 3600, step_size)):
        
        print('finding scores in step ' +  str(step + 1) + ' of ' + str(3600/step_size))
        
        test_dataset = Mouse_Dataset(mk_test_dat[:,t:t+step_size,:,:],
                                     sal_test_dat[:,t:t+step_size,:,:])
        
        mk_num_images = mk_test_dat[:,t:t+step_size,:,:].shape[0]*mk_test_dat[:,t:t+step_size,:,:].shape[1]
        sal_num_images = sal_test_dat[:,t:t+step_size,:,:].shape[0]*sal_test_dat[:,t:t+step_size,:,:].shape[1]
        test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle=False)

        y_true, y_pred, _ = test(test_dataloader, model, criterion)
        
        metrics['accuracy'][step].append(accuracy_score(y_true, y_pred))
        metrics['precision'][step].append(precision_score(y_true, y_pred, average='binary'))
        metrics['recall'][step].append(recall_score(y_true, y_pred, average='binary'))
        metrics['f1_score'][step].append(f1_score(y_true, y_pred, average='binary'))
        metrics['roc_auc_score'][step].append(roc_auc_score(y_true, y_pred))
        metrics['confusion_matrix'][step].append(confusion_matrix(y_true, y_pred))
        
        mk_avg_cam, sal_avg_cam, mk_avg_img, sal_avg_img = get_CAMs(model, test_dataset, mk_num_images = mk_num_images, sal_num_images=sal_num_images, cam_extractor=Xgrad)
       
        # plot_CAMs(mk_avg_cam, sal_avg_cam, mk_avg_img, sal_avg_img)
        
        mk_avg_CAMs.append(mk_avg_cam)
        sal_avg_CAMs.append(sal_avg_cam)
        
    mkCAMS = torch.stack(mk_avg_CAMs).numpy()
    salCAMS = torch.stack(sal_avg_CAMs).numpy()
    
    CAMS['MK'][str(train_mk_indices)].append(mkCAMS)
    CAMS['Saline'][str(train_sal_indices)].append(salCAMS)
        
# Save the dictionary 
np.save('mk_results/mk1_5/validation_metrics' + str(step_size), metrics)

np.save('mk_results/mk1_5/validation_CAMS' + str(step_size), CAMS)


import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from normalize import normalize_fUSI_bunch
from dataset import Mouse_Dataset
from timm.models.vision_transformer import VisionTransformer
from trainer import train, test
import random
from vit_rollout import VITAttentionRollout
import torch.nn.functional as F


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

attention_maps = {
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
    attention_maps['MK'][str(train_mk_indices)] = []
    attention_maps['Saline'][str(train_sal_indices)] = []

for train_mk_indices, train_sal_indices in training_options:
    train_dataset = Mouse_Dataset(mk_dat[train_mk_indices, 3300:, :, :], sal_dat[train_sal_indices, 3300:, :, :])

    mk_test_dat = np.delete(mk_dat, train_mk_indices, axis=0)
    sal_test_dat = np.delete(sal_dat, train_sal_indices, axis=0)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print('Training Vision Transformer!')
    model = VisionTransformer(
        img_size=(91, 128),  # Keep the same input size
        patch_size=(7, 8),   # Keep the same patch size
        embed_dim=256,       # Reduced embedding dimension (similar to CNN channels at the end)
        depth=4,             # Reduced number of transformer blocks
        num_heads=4,         # Reduced number of attention heads
        num_classes=2        # Keep the same number of output classes
    )
    
    model.patch_embed.proj = nn.Conv2d(
        in_channels=1,       # Single input channel
        out_channels=model.embed_dim,  # Matches the reduced embedding dimension
        kernel_size=(7, 8),  # Matches the patch size
        stride=(7, 8)        # Matches the patch size
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(3):
        train(train_dataloader, model, criterion, optimizer)
        
    model.eval()
    mk_attention, sal_attention = [], []
    
    grad_rollout = VITAttentionRollout(model, discard_ratio=0.9, head_fusion='max')

    for step, t in enumerate(np.arange(0, 3600, step_size)):
        print(f'Finding scores in step {step + 1} of {3600 // step_size}')

        test_dataset = Mouse_Dataset(mk_test_dat[:, t:t+step_size, :, :], sal_test_dat[:, t:t+step_size, :, :])
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        y_true, y_pred, outputs = test(test_dataloader, model, criterion)

        metrics['accuracy'][step].append(accuracy_score(y_true, y_pred))
        metrics['precision'][step].append(precision_score(y_true, y_pred, average='binary'))
        metrics['recall'][step].append(recall_score(y_true, y_pred, average='binary'))
        metrics['f1_score'][step].append(f1_score(y_true, y_pred, average='binary'))
        metrics['roc_auc_score'][step].append(roc_auc_score(y_true, y_pred))
        metrics['confusion_matrix'][step].append(confusion_matrix(y_true, y_pred))


        with torch.no_grad():
            dat_mk = torch.from_numpy(mk_test_dat[:, t:t+step_size, :, :]).float().flatten(end_dim = 1).unsqueeze(1)
            dat_sal = torch.from_numpy(sal_test_dat[:, t:t+step_size, :, :]).float().flatten(end_dim = 1).unsqueeze(1)
            
            mask_mk = grad_rollout(dat_mk)
            mask_sal = grad_rollout(dat_sal)
            
            mk_cam = F.interpolate(torch.from_numpy(mask_mk).unsqueeze(0).unsqueeze(0).float(), size=(91, 128), mode='bilinear', align_corners=True)[0,0,:,:]
            sal_cam = F.interpolate(torch.from_numpy(mask_sal).unsqueeze(0).unsqueeze(0).float(), size=(91, 128), mode='bilinear', align_corners=True)[0,0,:,:]
            # Process and save attention maps
            mk_attention.append(mk_cam)
            sal_attention.append(sal_cam)

    attention_maps['MK'][str(train_mk_indices)].append(np.array(mk_attention))
    attention_maps['Saline'][str(train_sal_indices)].append(np.array(sal_attention))

np.save('mk_results/mk1_5/validation_metrics_vit' + str(step_size), metrics)
np.save('mk_results/mk1_5/validation_attention_maps_vit' + str(step_size), attention_maps)

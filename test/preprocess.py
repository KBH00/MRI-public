import numpy as np
import polars as pl
import sys
import time
import yaml
from sklearn.model_selection import KFold
from loss import SevereLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from transformers import get_cosine_schedule_with_warmup

di = './Data'
device = torch.device('cuda')

tb_global = time.time()

train = pl.read_csv(di + '/train.csv',
                    schema_overrides={'study_id': str, 'series_id': str})
columns = train.columns[1:]

print('spinal:      ', columns[:5][:2], '...');     assert all(['spinal' in c for c in columns[:5]])
print('foraminal:   ', columns[5:15][:2], '...');   assert all(['foraminal' in c for c in columns[5:15]])
print('subarticular:', columns[15:25][:2], '...');  assert all(['subarticular' in c for c in columns[15:25]])

weight_map = {'Normal/Mild': 1,
              'Moderate': 2,
              'Severe': 4,
              None: 0}

w_sum = [0, ] * 4

for r in train.iter_rows():
    w = np.array([weight_map[x] for x in r[1:]])  # array[int] (25, )
    assert len(w) == 25

    w_sum[0] += w[:5].sum()    # spinal
    w_sum[1] += w[5:15].sum()  # foraminal
    w_sum[2] += w[15:25].sum() # subarticular 
    w_sum[3] += w[:5].max()    # any_severe_spinal

for k in range(4):
    w_sum[k] /= len(train)

w_sum
debug = False  # Train only 2 epochs if True

cfg = yaml.safe_load("""
data:
  image_size_in: 224

model:
  encoder: convnext_tiny.in12k_ft_in1k

kfold:
  k: 5
  folds: [0, ]

train:
  lr: 1e-4
  epochs: 20
  weight_decay: 1e-4   # these two are negligibly weak.
  max_grad_norm: 1000  # Maybe stronger are better for transformer models

validate:
  every_n_epoch: 1

loader:
  batch_size: 8
  num_workers: 2
""")

weight_decay = float(cfg['train']['weight_decay'])
max_grad_norm = cfg['train']['max_grad_norm']
val_every = cfg['validate']['every_n_epoch']

# Criterion
criterion = SevereLoss(temperature=0)
# criterion = nn.CrossEntropyLoss()       # for standard unweighted cross entropy

print('Criterion:', criterion)

# KFold
study_ids = [sid for sid, d in datad.items() if d['filenames'] is not None]
df = pl.DataFrame({'study_id': study_ids})

nfolds = cfg['kfold']['k']
folds = cfg['kfold']['folds']  # list[int]
kfold = KFold(n_splits=nfolds, shuffle=True, random_state=42)
print('Folds', folds, '/', nfolds)


#
# Training loop
#
study_ids = [sid for sid, d in datad.items() if d['filenames'] is not None]
df = pl.DataFrame({'study_id': study_ids})

for ifold, (idx_train, idx_val) in enumerate(kfold.split(df)):
    if ifold not in folds:
        continue
        
    # Data
    ds_train = Dataset(df[idx_train], datad, cfg, pick='random', augment=True)
    ds_val =   Dataset(df[idx_val],   datad, cfg, pick='middle')

    loader_train = ds_train.loader(cfg, shuffle=True, drop_last=True)
    loader_val = ds_val.loader(cfg)

    nbatch = len(loader_train)

    # Model
    model = Model(cfg, pretrained=False)  # Just use pretrained=True with internet
    
    # Read pretrained model because internet=off
    model_filename = '/kaggle/input/rsna2024-public/weights/model_pretrained.pytorch'
    model.load_state_dict(torch.load(model_filename))  
    # model.encoder.head.fc.reset_parameters()  # random initialize the classification head

    model.to(device)
    model.train()

    # Optimizer
    lr = float(cfg['train']['lr'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    epochs = 2 if debug else cfg['train']['epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                    num_warmup_steps=nbatch,
                    num_training_steps=(epochs * nbatch))

    print('%d epochs' % epochs)

    # n-epoch loop
    tb = time.time()
    dt_val = 0
    loss_sum, n_sum = 0, 0
    
    print('Epoch  loss          score   lr      time')
    for iepoch in range(epochs):
        for ibatch, d in enumerate(loader_train):
            x = d['x'].to(device)  # input image
            y = d['y'].to(device)  # segmentation label
            batch_size = len(x)

            optimizer.zero_grad()

            # Predict
            y_pred = model(x)      # (batch_size, 3, 25)
            loss = criterion(y_pred, y)

            # Backpropagate
            loss.backward()
            n_sum += batch_size
            loss_sum += batch_size * loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            scheduler.step()
        
        # Validation
        if (iepoch + 1) % val_every == 0:
            loss_train = loss_sum / n_sum

            val = evaluate(model, loader_val)
            dt_val += val['dt']
            
            lr = optimizer.param_groups[0]['lr']

            dt = time.time() - tb
            print('%3d %7.4f %7.4f  %.4f  %5.1e %.2f %.2f min' % (iepoch + 1,
                  loss_train, val['loss'], val['score'],
                  lr, dt_val / 60, dt / 60))

            loss_sum, n_sum = 0, 0

    # Save model
    model.to('cpu')
    model.eval()
    ofilename = 'model%d.pytorch' % ifold
    torch.save(model.state_dict(), ofilename)

    print(ofilename, 'written')


if debug:
    print('\nDebug %r: train only %d epochs.' % (debug, epochs))
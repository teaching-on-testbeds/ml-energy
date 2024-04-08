:::{.cell}
## Visualizing Energy Time Tradeoff
:::

:::{.cell .code}
```
import argparse
import os
import random
import time
from enum import Enum
```
:::

:::{.cell .code}
```
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
```
:::

:::{.cell .code}
```
from zeus.monitor import ZeusMonitor
from zeus.optimizer import GlobalPowerLimitOptimizer
from zeus.optimizer.power_limit import MaxSlowdownConstraint
from zeus.util.env import get_env
```
:::

:::{.cell}
## Loading Data
First, download the imagenet dataset
:::

:::{.cell .code}
```
#Downloading the train dataset
!wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
#Downloading the validataion dataset
!wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
#Download pytorch's official script to pre-process data
!wget https://raw.githubusercontent.com/pytorch/examples/main/imagenet/extract_ILSVRC.sh
```
:::

:::{.cell .code}
```
!. ./extract_ILSVRC.sh
```
:::

:::{.cell}
Define loaders for train and val data
:::

:::{.cell .code}
```
traindir = os.path.join('./', "train")
valdir = os.path.join('./', "val")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```
:::

:::{.cell .code}
```
train_dataset = datasets.ImageFolder(traindir,transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize,]),)

val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,]),)
```
:::

:::{.cell .code}
```
train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
)
```
:::


:::{.cell}
## Define the model
:::

:::{.cell .code}
```
ARCH = 'resnet50' #Defining the architecture to be used
model = models.__dict__[args.arch]()

torch.cuda.set_device(args.gpu)
model.cuda(args.gpu)
```
:::


:::{.cell}
Set the criteria to Cross-Entropy loss:
:::


:::{.cell .code}
```
criterion = nn.CrossEntropyLoss().cuda(args.gpu)
optimizer = torch.optim.Adadelta(
    model.parameters(),
    lr = 0.1
)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
```
:::


:::{.cell}
We'll now train the model; We stop the training when either a minimum training accuracy is reached or the max number of epochs have taken place.
:::

:::{.cell .code}
```
MAX_EPOCHS = 100
MIN_ACC = 0.65
```
:::


:::{.cell .code}
```
monitor = ZeusMonitor(gpu_indices=[0])
train_acc,train_samples = 0,0

for epoch in range(MAX_EPOCHS):

    ##Train the model for one epoch
    model.train()

    for i, (images, target) in enumerate(train_loader):

        # Load data to GPU
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # measure data loading time

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1))
        train_accuracy = (train_accuracy*train_samples + acc1*images.shape[0])/(train_samples+images.shape[0])
        train_samples += images.shape[0]
        
        losses.update(loss.item(), images.size(0))            

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(train_accuracy>MIN_ACC):
        break

    scheduler.step()

    measurement = monitor.end_window("model_train")
```
:::

:::{.cell}
# Plotting Energy Consumption
:::

:::{.cell .code}
```
energy = measurement.total_energy
time = measurement.time

plt.scatter(energy, time)
plt.xlabel('Training Time (s)')
plt.ylabel('Energy Consumption (J)')
```
:::
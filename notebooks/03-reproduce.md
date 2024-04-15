:::{.cell}
## Visualizing Energy Time Tradeoff

In this notebook, we'll attempt to visualize the relation of energy consumption during training and training time with varying batch size and GPU power limit. We reproduce Fig. 16 (d) of the [Zeus](https://www.usenix.org/system/files/nsdi23-you.pdf) paper to study is relation.
:::

:::{.cell .code}
```
import os
import time
import random
from matplotlib import pyplot as plt
```
:::

:::{.cell .code}
```
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pynvml
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

### Loading the train and test Data
We would be using the Imagenet dataset for this experiment. The total datasize of imagenet is approximately 150Gbs. Downloading this set on the server takes a significant amount of time (4 - 6 hours) so it is advised to start with the notebook early.
:::

:::{.cell .code}
```
#Downloading the train dataset (Takes ~ 4 hours)
!wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```
:::

:::{.cell .code}
```
#Downloading the validataion dataset (Takes ~ 6 minutes)
!wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
```
:::

:::{.cell .code}
```
#Downloading the imagenet devkit dataset for pytorch to pre-process dataset (Takes ~ 10 seconds)
!wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
```
:::

:::{.cell}
Define data tranformations that will be applied to the training and test dataset. We perform the following tranformations:

- Cropping each image to 224*224
- Augmenting the dataset with flipping images horizontly
- Tranforming image array to a tensor for it to be torch compatible
- Normalize the train and test dataet
:::

:::{.cell .code}
```
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


transformations = [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize,]
```
:::

:::{.cell}
Now load the downloaded train and test dataset as a torch dataset
:::

:::{.cell .code}
```
train_dataset = torchvision.datasets.ImageNet('./', split = 'train', tranform = transformations)

valid_dataset = torchvision.datasets.ImageNet('./', split = 'test', tranform = transformations)
```
:::


:::{.cell}

### Define the Neural Network
We will use the Resnet-50 architecture as our neural network for this experiement. Follow these steps to load the pre-defined Resnet-50 architecture from the torchvision.models module
:::

:::{.cell .code}
```
#Setting the device to CUDA if GPU is available for super-fast training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
```
:::

:::{.cell .code}
```
ARCH = 'resnet50' #Defining the architecture to be used
model = models.__dict__[ARCH]() #Loading the architecture

torch.cuda.set_device(torch.cuda.device(device))
model.cuda(device) #Placing the model in the GPU, if available
```
:::

:::{.cell}

### Train the Neural Network

To train the network, we have to select an optimizer and a loss function.

Since this is a multi-class classification problem, we select the cross entropy loss.

We will also choose an optimizer (Adadelta) as defined in Section 6.1 of the [Zeus](https://www.usenix.org/system/files/nsdi23-you.pdf) paper
:::

:::{.cell .code}
```
#Set the criteria to Cross-Entropy loss
criterion = nn.CrossEntropyLoss().cuda(args.gpu)
optimizer = torch.optim.Adadelta(
    model.parameters(),
    lr = 0.1
)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
```
:::

:::{.cell}
To pass data to our model, we will prepare a DataLoader - this will iterate over the data and “batch” it for us according to the batch size we specify.
:::

:::{.cell .code}
```
#Loading the torch datasets as dataloaders to pass in the model; Using a default batch size of 128; shuffle for training, not for validation
BATCH_SIZE = 128

train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
val_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
```
:::



:::{.cell}
Now we are ready to define our training function:

-   Get a batch of training data from the `train_loader`.
-   Zero the gradients of the `optimizer`. (This is necessary because by default, they accumulate.)
-   Do a forward pass on the batch of training data.
-   Use the predictions from this forward pass to compute the loss.
-   Then, do a backwards pass where we compute the gradients.
-   Update the weights of the optimizer using these gradients.

We stop the training when either a minimum validation accuracy is reached or the max number of epochs have taken place.
:::

:::{.cell .code}
```
def train_one_epoch(data_loader):

    #Put the model in training mode
    model.train()

    for i, (images, target) in enumerate(data_loader):

        # Load data to GPU
        images = images.cuda(device, non_blocking=True)
        target = target.cuda(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
:::

:::{.cell}
We will also define a function for evaluating the model without training:
:::


:::{.cell .code}
```
#First define an accuracy function to get the accuracy of a batch prediction
def accuracy(predicted,actual):
    _, predictions = torch.max(predicted,dim=1)
    return torch.tensor(torch.sum(predictions==actual).item()/len(predictions))


def eval_model(data_loader):
    eval_accuracy = 0
    running_samples = 0

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation for faster computation/reduced memory
    with torch.no_grad():

      for i, (images, target) in enumerate(data_loader):
          # Every data instance is an X, y pair
          images = images.cuda(device, non_blocking=True)
          target = target.cuda(device, non_blocking=True)

          # Forward pass makes predictions for this batch
          output = model(images)

          # Compute the accuracy
          accuracy_batch = accuracy(output, target)
          eval_accuracy = (eval_accuracy*running_samples + accuracy_batch*images.shape[0])/(running_samples+images.shape[0])
          running_samples += images.shape[0]
    
    return eval_accuracy
```
:::

:::{.cell}
Now, we will loop over epochs, train the model for one epoch, and then evaluate its performance on the validation data at the end of each epoch.
:::


:::{.cell .code}
```
n_max_iter = 1
MIN_VALIDATION_ACC = 0.65 #Stop training if validation accuracy reaches atleast MIN_VALIDATION_ACC
metrics = {'validation_acc': []}
train_acc,train_samples = 0,0

#Training for one epoch
monitor = ZeusMonitor(gpu_indices=[0])

try:
    monitor.begin_window("model_train")
# if the last measurement window is still running
except ValueError:
    _ = monitor.end_window("model_train")
    monitor.begin_window("model_train")

for epoch in range(n_max_iter):

    ##Train the model for one epoch

    # Train on training data
    train_one_epoch(train_loader)

    # Evaluate on validation data
    val_accuracy = eval_model(val_loader)
    metrics['validation_acc'].append(val_accuracy)

    print(f'Epoch {epoch+1}/{n_epochs} - Val_Accuracy: {val_accuracy:.4f}')


    if(val_accuracy>=MIN_VALIDATION_ACC):
        break

measurement = monitor.end_window("model_train")
```
:::

:::{.cell .code}
```
#Checkout enegy and time required for one epoch
print("Measured time (s)  :" , measurement.time)
print("Measured energy (J):" , measurement.total_energy)
```
:::


:::{.cell}
### Reproducing the Figure

Now, we run the above loop to `n_max_iter = 100` with varying batch sizes and GPU power limit and store the model performance and energy metrics
:::


:::{.cell .code}
```
#Select all the batch size and GPU power limits to test
batch_size_list = [2**bts for bts in range(3,11)] #Sizes ranging from 8 to 1024 increasing in powers of 2

power_limit_list = [powlimit for powlimit in range(100,251,30)] #Power limit ranging from 100W to 350W with increments of 30W

#We now sample `k` random pair of batch_sizes and power limits

k = 5

params = sorted(random.sample(list(zip(batch_size_list, power_limit_list)), k))
```
:::

:::{.cell .code}
```
n_max_iter = 100
MIN_VALIDATION_ACC = 0.65 #Stop training if validation accuracy reaches atleast MIN_VALIDATION_ACC
metrics = {'validation_acc': [],'time':[],'energy':[],'power_limit':[],'batch_size':[],'num_epochs'[]}
train_acc,train_samples = 0,0

#Training for one epoch
monitor = ZeusMonitor(gpu_indices=[0])



for (BATCH_SIZE, power_limit) in params:

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    pynvml.nvmlDeviceSetPowerManagementLimit(pynvmml.nvmlDeviceGetHandleByIndex(0),power_limit*1000) #Setting the power limit


    try:
        monitor.begin_window("model_train")
    # if the last measurement window is still running
    except ValueError:
        _ = monitor.end_window("model_train")
        monitor.begin_window("model_train")
    for epoch in range(n_max_iter):

        ##Train the model for one epoch

        # Train on training data
        train_one_epoch(train_loader)

        # Evaluate on validation data
        val_accuracy = eval_model(val_loader)

        print(f'Epoch {epoch+1}/{n_epochs} - Val_Accuracy: {val_accuracy:.4f}')


        if(val_accuracy>=MIN_VALIDATION_ACC):
            break

    measurement = monitor.end_window("model_train")

    #Saving the metrics
    metrics['validation_acc'].append(val_accuracy)
    metrics['time'].append(measurement.time)
    metrics['energy'].append(measurement.total_energy)
    metrics['power_limit'].append(power_limit)
    metrics['batch_size'].append(BATCH_SIZE)
    metrics['num_epocs'].append(epoch+1)
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
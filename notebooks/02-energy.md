:::{.cell}
## Energy and time to train a neural network
In the previous notebook, we trained a model with a specific learning rate and batch size. However, the choice of training hyperparameters can have a substantial impact on the time and energy it takes to train our model. In this notebook, we will train a series of models all to the same validation accuracy "target", but with different training hyperparameters. For each model, we will consider:

-   how much *time* it takes to achieve that accuracy target (“time to accuracy”)
-   how much *energy* it takes to achieve that accuracy target (“energy to accuracy”)
-   and the *test accuracy* for the model, given that it is trained to the specified validation accuracy target
:::

:::{.cell .code}
```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
:::


:::{.cell .code}
```
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
```
:::

:::{.cell}
### Loading Data

Here, we'll load the processed data defined in the previous notebook
:::

:::{.cell .code}
```
Xtr_scale = np.load('instrument_dataset/uiowa_std_scale_train_data.npy')
ytr = np.load('instrument_dataset/uiowa_permuted_train_labels.npy')
Xts_scale = np.load('instrument_dataset/uiowa_std_scale_test_data.npy')
yts = np.load('instrument_dataset/uiowa_test_labels.npy')
```
:::


:::{.cell .code}
```
nh = 256 #Number of units for the hidden layer of the neural network
```
:::

:::{.cell}
### Energy consumption

To do this, first we will need some way to measure the energy used to train the model. We will use [Zeus](https://ml.energy/zeus/overview/), a Python package developed by researchers at the University of Michigan, to measure the GPU energy consumption.
:::

:::{.cell}
Install the zeus-ml package, we will be using this package to measure energy comsumption of the GPU
:::

:::{.cell .code}
```
!pip install zeus==0.8.2
```
:::

:::{.cell}
Import the zeus-ml package, start an instance of a monitor, specifying the GPU that it should monitor
:::


:::{.cell .code}
```
from zeus.monitor import ZeusMonitor

monitor = ZeusMonitor()
```
:::


:::{.cell}
When you want to measure GPU energy usage, you will:

-   start a “monitoring window”
-   do your GPU-intensive computation (e.g. call `model.fit`)
-   stop the “monitoring window”

and then you can get the time and total energy used by the GPU in the monitoring window.

Try it now :
:::

:::{.cell .code}
```
model = Sequential()
model.add(Input((Xtr_scale.shape[1],)))
model.add(Dense(nh, activation = 'sigmoid'))
model.add(Dense(len(np.unique(ytr)), activation = 'softmax'))

opt = optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer = opt, loss = loss_fn, metrics = ['accuracy'])
```
:::

:::{.cell .code}
```
try:
    monitor.begin_window("test")
# if the last measurement window is still running
except ValueError:
    _ = monitor.end_window("test")
    monitor.begin_window("test")

model.fit(Xtr_scale, ytr, epochs=5)
measurement = monitor.end_window("test")
print("Measured time (s)  :" , measurement.time)
print("Measured energy (J):" , measurement.total_energy)
```
:::

:::{.cell}
### `TrainToAccuracy` callback

Next, we need a way to train a model until we achieve our desired validation accuracy. We will [write a callback function](https://www.tensorflow.org/guide/keras/writing_your_own_callbacks) following these specifications:

-   It will be called `TrainToAccuracy` and will accept two arguments: a `threshold` and a `patience` value.
-   If the model’s validation accuracy is higher than the `threshold` for `patience` epochs in a row, stop training.
-   In the `on_epoch_end` function, which will be called at the end of every epoch during training, you should get the current validation accuracy using `currect_acc = logs.get("val_accuracy")`. Then, set `self.model.stop_training = True` if the condition above is met.
-   The default values of `threshold` and `patience` are given below, but other values may be passed as arguments at runtime.

Then, when you call `model.fit()`, you will add the `TrainToAccuracy` callback as in

    callbacks=[TrainToAccuracy(threshold=0.98, patience=5)]
:::

:::{.cell .code}
```
# TODO - write a callback function
```
:::

:::{.cell}
Try it! run the following cell to test your `TrainToAccuracy` callback. 
:::

:::{.cell .code}
```
model.fit(Xtr_scale, ytr, epochs=100, validation_split = 0.2, callbacks=[TrainToAccuracy(threshold=0.98, patience=5)])
```
:::

:::{.cell}
Your model shouldn’t *really* train for 100 epochs - it should stop training as soon as 98% validation accuracy is achieved for 5 epochs in a row! (Your “test” is not graded, you may change the `threshold` and `patience` values in this “test” call to `model.fit` in order to check your work.)

Note that since we are now using the validation set performance to *decide* when to stop training the model, we are no longer “allowed” to pass the test set as `validation_data`. The test set must never be used to make decisions during the model training process - only for evaluation of the final model. Instead, we specify that 20% of the training data should be held out as a validation set, and that is the validation accuracy that is used to determine when to stop training.
:::


:::{.cell}
### See how TTA/ETA varies with learning rate, batch size

Now, you will repeat your model preparation and fitting code - with your new `TrainToAccuracy` callback - but in a loop. First, you will iterate over different learning rates. Consider the following learning rates for this experiment.

```python
[0.0001, 0.001, 0.01, 0.1]
```

In each iteration of each loop, you will prepare a model (with the appropriate training hyperparameters) and train it until:

-   either it has achieved **0.98 accuracy for 3 epoches in a row** on a 20% validation subset of the training data,
-   or, it has trained for 500 epochs

whichever comes FIRST.
:::

:::{.cell}
For each model, you will record:

-   the training hyperparameters (learning rate, batch size)
-   the number of epochs of training needed to achieve the target validation accuracy
-   the accuracy on the *test* data (not the validation data!). After fitting the model, use `model.evaluate` and pass the scaled *test* data to get the test loss and test accuracy
-   The GPU energy and time to train the model to the desired validation accuracy, as computed by a `zeus-ml` measurement window that starts just before `model.fit` and ends just after `model.fit`.
:::

:::{.cell .code}
```

# TODO - iterate over learning rates and get TTA/ETA

# default learning rate and batch size -
batch_size = 128

metrics_vs_lr = []
for lr in [0.0001, 0.001, 0.01, 0.1]:
    # TODO - set up model, including appropriate optimizer hyperparameters

    # start measurement
    try:
        monitor.begin_window("model_train")
    # if the last measurement window is still running
    except ValueError:
        _ = monitor.end_window("model_train")
        monitor.begin_window("model_train")


    # TODO - fit model on (scaled) training data
    # until specified validation accuracy is achieved (don't use test data!)
    # but stop after 500 epochs even if validation accuracy is not achieved


    # end measurement
    measurement = monitor.end_window("model_train")

    # TODO - evaluate model on (scaled) test data


    # save results in a dictionary
    model_metrics = {
       'batch_size': ,
       'learning_rate': ,
       'epochs': ,
       'test_accuracy': ,
       'total_energy': ,
       'train_time': 
    }

    # TODO - append model_metrics dictionary to the metrics_vs_lr list
```
:::

:::{.cell}
Next, you will visualize the results.

Create a figure with four subplots. In each subplot, create a bar plot with learning rate on the horizontal axis and (1) Time to accuracy, (2) Energy to accuracy, (3) Test accuracy, (4) Epochs, on the vertical axis on each subplot, respectively. Use an appropriate vertical range for each subplot. Label all axes.
:::

:::{.cell .code}
```
# TODO - visualize effect of varying learning rate, when training to a target accuracy
```
:::


:::{.cell}
**Comment on the results**: Given that the model is trained to a target validation accuracy, what is the effect of the learning rate on the training process?
:::

:::{.cell}
Now, you will repeat, with a loop over different batch sizes. Consider the following batch sizes - 

```python
[64, 128, 256, 512, 1024, 2048, 4096, 8192]
```
:::

:::{.cell .code}
```
# TODO - iterate over batch size and get TTA/ETA

# default learning rate and batch size -
lr = 0.001
batch_size = 128
metrics_vs_bs = []

for batch_size in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:

    # TODO - set up model, including appropriate optimizer hyperparameters

    # start measurement
    try:
        monitor.begin_window("model_train")
    # if the last measurement window is still running
    except ValueError:
        _ = monitor.end_window("model_train")
        monitor.begin_window("model_train")


    # TODO - fit model on (scaled) training data
    # until specified validation accuracy is achieved (don't use test data!)
    # but stop after 500 epochs even if validation accuracy is not achieved

    # end measurement
    measurement = monitor.end_window("model_train")

    # TODO - evaluate model on (scaled) test data

    # save results in a dictionary
    model_metrics = {
       'batch_size': ,
       'learning_rate': ,
       'epochs': ,
       'test_accuracy': ,
       'total_energy': ,
       'train_time': 
    }

    # TODO - append model_metrics dictionary to the metrics_vs_bs list
```
:::

:::{.cell}
Next, you will visualize the results.

Create a figure with four subplots. In each subplot, create a bar plot with batch size on the horizontal axis and (1) Time to accuracy, (2) Energy to accuracy, (3) Test accuracy, (4) Epochs, on the vertical axis on each subplot, respectively. Use an appropriate vertical range for each subplot. Label all axes.
:::

:::{.cell .code}
```
# TODO - visualize effect of varying batch size, when training to a target accuracy

```
:::

:::{.cell}
**Comment on the results**: Given that the model is trained to a target validation accuracy, what is the effect of the batch size on the training process in this example? What do you observe about how time and energy per epoch and number of epochs required varies with batch size?
:::
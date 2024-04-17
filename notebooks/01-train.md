:::{.cell}
## Building a Neural Network Classifier
:::

:::{.cell .code}
```
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
:::

:::{.cell}
## Loading Data

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

:::{.cell}
## Building the classification model
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
Create a neural network `model` with:

-   `nh=256` hidden units in a single dense hidden layer
-   `sigmoid` activation at hidden units
-   select the input and output shapes, and output activation, according to the problem requirements.
:::

:::{.cell .code}
```
# TODO - construct the model
nh = 256
model = Sequential()
```
:::

:::{.cell}
Print the model summary.
:::

:::{.cell .code}
```
# show the model summary
model.summary()
```
:::

:::{.cell .code}
```
# you can also visualize the model with
tf.keras.utils.plot_model(model, show_shapes=True)
```
:::

:::{.cell}
Create an optimizer and compile the model. Select the appropriate loss function for this multi-class classification problem, and use an accuracy metric. For the optimizer, use the Adam optimizer with a learning rate of 0.001
:::

:::{.cell .code}
```
# TODO - create optimizer and compile the model
opt = 
loss_fn = 

model.compile()
```
:::

:::{.cell}
Fit the model for 10 epochs using the scaled data for both training and validation, and save the training history in \`hist\`.

Use the `validation_data` option to pass the *test* data. (This is OK because we are not going to use this data as part of the training process, such as for early stopping - we’re just going to compute the accuracy on the data so that we can see how training and test loss changes as the model is trained.)

Use a batch size of 128. Your final accuracy should be greater than 99%.
:::

:::{.cell .code}
```
# TODO - fit model and save training history
n_epochs = 10

hist = model.fit()
```
:::

:::{.cell}
Plot the training and validation accuracy saved in `hist.history` dictionary, on the same plot. This gives one accuracy value per epoch. You should see that the validation accuracy saturates around 99%. After that it may “bounce around” a little due to the noise in the stochastic mini-batch gradient descent.

Make sure to label each axis, and each series (training vs. validation/test).
:::

:::{.cell .code}
```
# TODO - plot the training and validation accuracy in one plot
```
:::

:::{.cell}
Plot the training and validation loss values saved in the `hist.history` dictionary, on the same plot. You should see that the training loss is steadily decreasing. Use the [`semilogy` plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.semilogy.html) so that the y-axis is log scale.

Make sure to label each axis, and each series (training vs. validation/test).
:::

:::{.cell .code}
```
# TODO - plot the training and validation loss in one plot
```
:::

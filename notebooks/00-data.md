:::{.cell}
# Load data for music classification

**Note**: This experiment is designed to run on a GPU runtime.

In this assignment, we will look at an audio classification problem. Given a sample of music, we want to determine which instrument (e.g. trumpet, violin, piano) is playing.

*This assignment is closely based on one by Sundeep Rangan, from his [IntroML GitHub repo](https://github.com/sdrangan/introml/).*
:::


:::{.cell .code}
```
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
```
:::

:::{.cell}
## Audio Feature Extraction with Librosa

The key to audio classification is to extract the correct features. The `librosa` package in python has a rich set of methods for extracting the features of audio samples commonly used in machine learning tasks, such as speech recognition and sound classification.
:::

:::{.cell .code}
```
import librosa
import librosa.display
import librosa.feature
```
:::

:::{.cell}
In this lab, we will use a set of music samples from the website:

http://theremin.music.uiowa.edu

This website has a great set of samples for audio processing.

We will use the `wget` command to retrieve one file to our Google Colab storage area. (We can run `wget` and many other basic Linux commands in Colab by prefixing them with a `!` or `%`.)
:::

:::{.cell .code}
```
!wget "http://theremin.music.uiowa.edu/sound files/MIS/Woodwinds/sopranosaxophone/SopSax.Vib.pp.C6Eb6.aiff"
```
:::

:::{.cell}
Now, if you click on the small folder icon on the far left of the Colab interface, you can see the files in your Colab storage. You should see the “SopSax.Vib.pp.C6Eb6.aiff” file appear there.

In order to listen to this file, we’ll first convert it into the `wav` format. Again, we’ll use a magic command to run a basic command-line utility: `ffmpeg`, a powerful tool for working with audio and video files.
:::

:::{.cell .code}
```
aiff_file = 'SopSax.Vib.pp.C6Eb6.aiff'
wav_file = 'SopSax.Vib.pp.C6Eb6.wav'

!ffmpeg -y -i $aiff_file $wav_file
```
:::

:::{.cell}
Now, we can play the file directly from the Jupyter Notebook interface. If you press the ▶️ button, you will hear a soprano saxaphone (with vibrato) playing four notes (C, C#, D, Eb).
:::

:::{.cell .code}
```
import IPython.display as ipd
ipd.Audio(wav_file)
```
:::

:::{.cell}
Next, use `librosa` command `librosa.load` to read the audio file with filename `audio_file` and get the samples `y` and sample rate `sr`.
:::

:::{.cell .code}
```
y, sr = librosa.load(aiff_file)
```
:::

:::{.cell}
Feature engineering from audio files is an entire subject in its own right. A commonly used set of features are called the Mel Frequency Cepstral Coefficients (MFCCs). These are derived from the so-called mel spectrogram, which is something like a regular spectrogram, but the power and frequency are represented in log scale, which more naturally aligns with human perceptual processing.

You can run the code below to display the mel spectrogram from the audio sample.

You can easily see the four notes played in the audio track. You also see the 'harmonics' of each notes, which are other tones at integer multiples of the fundamental frequency of each note.
:::

:::{.cell .code}
```
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
librosa.display.specshow(librosa.amplitude_to_db(S),
                         y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
```
:::

:::{.cell}
## Downloading the Data

Using the MFCC features described above, [Prof. Juan Bello](http://steinhardt.nyu.edu/faculty/Juan_Pablo_Bello) at NYU Steinhardt and his former PhD student Eric Humphrey have created a complete data set that can used for instrument classification. Essentially, they collected a number of data files from the website above. For each audio file, the segmented the track into notes and then extracted 120 MFCCs for each note. The goal is to recognize the instrument from the 120 MFCCs. The process of feature extraction is quite involved. So, we will just use their processed data.
:::

:::{.cell}
To retrieve their data, visit

<https://github.com/marl/dl4mir-tutorial/tree/master>

and note the password listed on that page. Click on the link for “Instrument Dataset”, enter the password, click on `instrument_dataset` to open the folder, and download it. (You can “direct download” straight from this site, you don’t need a Dropbox account.) Depending on your laptop OS and on how you download the data, you may need to “unzip” or otherwise extract the four `.npy` files from an archive.

Now create a new folder (named `instrument_dataset`) on the Chameleon server for storing the dataset.
:::

:::{.cell .code}
```
!mkdir instrument_dataset/
```
:::

:::{.cell}
Then, upload the files to Chameleon server inside the `instrument_dataset` folder: click on the folder icon on the left to see your storage, if it isn’t already open, and then click on “Upload”.

🛑 Wait until *all* uploads have completed. To check if all the files have been uploaded successfully, check the size of the `instrument_dataset` on the server using the following cell. If all uploads are successful, the folder size should be 75M. 🛑
:::

:::{.cell .code}
```
!du -sh instrument_dataset
```
:::

:::{.cell}
Then, load the files with:
:::

:::{.cell .code}
```
Xtr = np.load('instrument_dataset/uiowa_train_data.npy')
ytr = np.load('instrument_dataset/uiowa_train_labels.npy')
Xts = np.load('instrument_dataset/uiowa_test_data.npy')
yts = np.load('instrument_dataset/uiowa_test_labels.npy')
```
:::

:::{.cell}
Examine the data you have just loaded in:

-   How many training samples are there?
-   How many test samples are there?
-   What is the number of features for each sample?
-   How many classes (i.e. instruments) are there?

Write some code to find these values and print them.
:::

:::{.cell .code}
```
# TODO -  get basic details of the data
# compute these values from the data, don't hard-code them
n_tr    = 
n_ts    = 
n_feat  = 
n_class = 
```
:::


:::{.cell .code}
```
# now print those details
print("Num training= %d" % n_tr)
print("Num test=     %d" % n_ts)
print("Num features= %d" % n_feat)
print("Num classes=  %d" % n_class)
```
:::


:::{.cell .code}
```
# shuffle the training set
# (when loaded in, samples are ordered by class)
p = np.random.permutation(Xtr.shape[0])
Xtr = Xtr[p,:]
ytr = ytr[p]
```
:::

:::{.cell}
Then, standardize the training and test data, `Xtr` and `Xts`, by removing the mean of each feature and scaling to unit variance.

You can do this manually, or using `sklearn`'s [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

Although you will scale both the training and test data, you should make sure that both are scaled according to the mean and variance statistics from the *training data only*.

<small>Standardizing the input data can make the gradient descent work better, by making the loss function “easier” to descend.</small>
:::

:::{.cell .code}
```
from sklearn.preprocessing import StandardScaler
```
:::

:::{.cell .code}
```
scaler = StandardScaler()
```
:::


:::{.cell .code}
```
# TODO - Standardize the training and test data
Xtr_scale = 
Xts_scale = 
```
:::

:::{.cell}
Saving the standardized training and test data features for further use.
:::

:::{.cell .code}
```
np.save('instrument_dataset/uiowa_std_scale_train_data.npy',Xtr_scale)
np.save('instrument_dataset/uiowa_std_scale_test_data.npy',Xts_scale)
np.save('instrument_dataset/uiowa_permuted_train_labels.npy',ytr)
```
:::

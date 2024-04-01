# Assignment: Neural Networks for Music Classification

*Fraida Fund*

**TODO**: Edit this cell to fill in your NYU Net ID and your name:

-   **Net ID**:
-   **Name**:

⚠️ **Note**: This experiment is designed to run on a Google Colab **GPU** runtime. You should use a GPU runtime on Colab to work on this assignment. You should not run it outside of Google Colab. However, if you have been using Colab GPU runtimes a lot, you may be alerted that you have exhausted the “free” compute units allocated to you by Google Colab. If that happens, you do not have to purchase compute units - use a CPU runtime instead, and modify the experiment as instructed for CPU-only runtime.

In this assignment, we will look at an audio classification problem. Given a sample of music, we want to determine which instrument (e.g. trumpet, violin, piano) is playing.

*This assignment is closely based on one by Sundeep Rangan, from his [IntroML GitHub repo](https://github.com/sdrangan/introml/).*


```python
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import time
%matplotlib inline
```

## Audio Feature Extraction with Librosa

The key to audio classification is to extract the correct features. The `librosa` package in python has a rich set of methods for extracting the features of audio samples commonly used in machine learning tasks, such as speech recognition and sound classification.


```python
import librosa
import librosa.display
import librosa.feature
```

In this lab, we will use a set of music samples from the website:

<http://theremin.music.uiowa.edu>

This website has a great set of samples for audio processing.

We will use the `wget` command to retrieve one file to our Google Colab storage area. (We can run `wget` and many other basic Linux commands in Colab by prefixing them with a `!` or `%`.)


```python
!wget "http://theremin.music.uiowa.edu/sound files/MIS/Woodwinds/sopranosaxophone/SopSax.Vib.pp.C6Eb6.aiff"
```

    --2024-03-31 20:51:29--  http://theremin.music.uiowa.edu/sound%20files/MIS/Woodwinds/sopranosaxophone/SopSax.Vib.pp.C6Eb6.aiff
    Resolving theremin.music.uiowa.edu (theremin.music.uiowa.edu)... 128.255.102.97
    Connecting to theremin.music.uiowa.edu (theremin.music.uiowa.edu)|128.255.102.97|:80... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://theremin.music.uiowa.edu/sound%20files/MIS/Woodwinds/sopranosaxophone/SopSax.Vib.pp.C6Eb6.aiff [following]
    --2024-03-31 20:51:29--  https://theremin.music.uiowa.edu/sound%20files/MIS/Woodwinds/sopranosaxophone/SopSax.Vib.pp.C6Eb6.aiff
    Connecting to theremin.music.uiowa.edu (theremin.music.uiowa.edu)|128.255.102.97|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1418242 (1.4M) [audio/aiff]
    Saving to: 'SopSax.Vib.pp.C6Eb6.aiff'
    
    SopSax.Vib.pp.C6Eb6 100%[===================>]   1.35M  3.37MB/s    in 0.4s    
    
    2024-03-31 20:51:30 (3.37 MB/s) - 'SopSax.Vib.pp.C6Eb6.aiff' saved [1418242/1418242]
    


Now, if you click on the small folder icon on the far left of the Colab interface, you can see the files in your Colab storage. You should see the “SopSax.Vib.pp.C6Eb6.aiff” file appear there.

In order to listen to this file, we’ll first convert it into the `wav` format. Again, we’ll use a magic command to run a basic command-line utility: `ffmpeg`, a powerful tool for working with audio and video files.


```python
aiff_file = 'SopSax.Vib.pp.C6Eb6.aiff'
wav_file = 'SopSax.Vib.pp.C6Eb6.wav'

!ffmpeg -y -i $aiff_file $wav_file
```

    ffmpeg version 4.2.2 Copyright (c) 2000-2019 the FFmpeg developers
      built with clang version 12.0.0
      configuration: --prefix=/Users/ktietz/demo/mc3/conda-bld/ffmpeg_1628925491858/_h_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_plac --cc=arm64-apple-darwin20.0.0-clang --disable-doc --enable-avresample --enable-gmp --enable-hardcoded-tables --enable-libfreetype --enable-libvpx --enable-pthreads --enable-libopus --enable-postproc --enable-pic --enable-pthreads --enable-shared --enable-static --enable-version3 --enable-zlib --enable-libmp3lame --disable-nonfree --enable-gpl --enable-gnutls --disable-openssl --enable-libopenh264 --enable-libx264
      libavutil      56. 31.100 / 56. 31.100
      libavcodec     58. 54.100 / 58. 54.100
      libavformat    58. 29.100 / 58. 29.100
      libavdevice    58.  8.100 / 58.  8.100
      libavfilter     7. 57.100 /  7. 57.100
      libavresample   4.  0.  0 /  4.  0.  0
      libswscale      5.  5.100 /  5.  5.100
      libswresample   3.  5.100 /  3.  5.100
      libpostproc    55.  5.100 / 55.  5.100
    [0;33mGuessed Channel Layout for Input Stream #0.0 : mono
    [0mInput #0, aiff, from 'SopSax.Vib.pp.C6Eb6.aiff':
      Duration: 00:00:16.07, start: 0.000000, bitrate: 705 kb/s
        Stream #0:0: Audio: pcm_s16be, 44100 Hz, mono, s16, 705 kb/s
    Stream mapping:
      Stream #0:0 -> #0:0 (pcm_s16be (native) -> pcm_s16le (native))
    Press [q] to stop, [?] for help
    Output #0, wav, to 'SopSax.Vib.pp.C6Eb6.wav':
      Metadata:
        ISFT            : Lavf58.29.100
        Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 44100 Hz, mono, s16, 705 kb/s
        Metadata:
          encoder         : Lavc58.54.100 pcm_s16le
    size=    1385kB time=00:00:16.07 bitrate= 705.6kbits/s speed=4.93e+03x    
    video:0kB audio:1384kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.005502%


Now, we can play the file directly from Colab. If you press the ▶️ button, you will hear a soprano saxaphone (with vibrato) playing four notes (C, C#, D, Eb).


```python
import IPython.display as ipd
ipd.Audio(wav_file)
```





<audio  controls="controls" >
    Your browser does not support the audio element.
</audio>




Next, use `librosa` command `librosa.load` to read the audio file with filename `audio_file` and get the samples `y` and sample rate `sr`.


```python
y, sr = librosa.load(aiff_file)
```

Feature engineering from audio files is an entire subject in its own right. A commonly used set of features are called the Mel Frequency Cepstral Coefficients (MFCCs). These are derived from the so-called mel spectrogram, which is something like a regular spectrogram, but the power and frequency are represented in log scale, which more naturally aligns with human perceptual processing.

You can run the code below to display the mel spectrogram from the audio sample.

You can easily see the four notes played in the audio track. You also see the 'harmonics' of each notes, which are other tones at integer multiples of the fundamental frequency of each note.


```python
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
librosa.display.specshow(librosa.amplitude_to_db(S),
                         y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
```


    
![png](output_17_0.png)
    


## Downloading the Data

Using the MFCC features described above, [Prof. Juan Bello](http://steinhardt.nyu.edu/faculty/Juan_Pablo_Bello) at NYU Steinhardt and his former PhD student Eric Humphrey have created a complete data set that can used for instrument classification. Essentially, they collected a number of data files from the website above. For each audio file, the segmented the track into notes and then extracted 120 MFCCs for each note. The goal is to recognize the instrument from the 120 MFCCs. The process of feature extraction is quite involved. So, we will just use their processed data.


```python
Xtr = np.load('instrument_dataset/uiowa_train_data.npy')
ytr = np.load('instrument_dataset/uiowa_train_labels.npy')
Xts = np.load('instrument_dataset/uiowa_test_data.npy')
yts = np.load('instrument_dataset/uiowa_test_labels.npy')
```

Examine the data you have just loaded in:

-   How many training samples are there?
-   How many test samples are there?
-   What is the number of features for each sample?
-   How many classes (i.e. instruments) are there?

Write some code to find these values and print them.


```python
# TODO -  get basic details of the data
# compute these values from the data, don't hard-code them
n_tr    = Xtr.shape[0]
n_ts    = Xts.shape[0]
n_feat  = Xtr.shape[1]
n_class = len(np.unique(ytr))
```


```python
# now print those details
print("Num training= %d" % n_tr)
print("Num test=     %d" % n_ts)
print("Num features= %d" % n_feat)
print("Num classes=  %d" % n_class)
```

    Num training= 66247
    Num test=     14904
    Num features= 120
    Num classes=  10



```python
# shuffle the training set
# (when loaded in, samples are ordered by class)
p = np.random.permutation(n_tr)
Xtr = Xtr[p,:]
ytr = ytr[p]
```

Then, standardize the training and test data, `Xtr` and `Xts`, by removing the mean of each feature and scaling to unit variance.

You can do this manually, or using `sklearn`'s [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). (For an example showing how to use a `StandardScaler`, you can refer to the notebook on regularization.)

Although you will scale both the training and test data, you should make sure that both are scaled according to the mean and variance statistics from the *training data only*.

<small>Standardizing the input data can make the gradient descent work better, by making the loss function “easier” to descend.</small>


```python
from sklearn.preprocessing import StandardScaler
```


```python
scaler = StandardScaler()
```


```python
# TODO - Standardize the training and test data
Xtr_scale = scaler.fit_transform(Xtr)
Xts_scale = scaler.transform(Xts)
```

Saving the standardized training and test data features for further use.


```python
np.save('instrument_dataset/uiowa_std_scale_train_data.npy',Xtr_scale)
np.save('instrument_dataset/uiowa_std_scale_test_data.npy',Xts_scale)
np.save('instrument_dataset/uiowa_permuted_train_labels.npy',ytr)
```
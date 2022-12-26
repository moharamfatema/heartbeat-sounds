# Heartbeat Sounds Classification and Segmentation

[![github-badge]][repo]

- [Heartbeat Sounds Classification and Segmentation](#heartbeat-sounds-classification-and-segmentation)
  - [The Dataset](#the-dataset)
  - [The Classification Model](#the-classification-model)
    - [Data Preprocessing](#data-preprocessing)
    - [Hyperparameters](#hyperparameters)
    - [Feed Forward Fully Connected Neural Network](#feed-forward-fully-connected-neural-network)
      - [Architecture 1](#architecture-1)
      - [Architecture 2](#architecture-2)
      - [Architecture 3 : Deeper](#architecture-3--deeper)
    - [Test Results For Feed Forward Fully Connected Neural Network](#test-results-for-feed-forward-fully-connected-neural-network)
      - [Model A](#model-a)
        - [Classification Report](#classification-report)
        - [Confusion Matrix](#confusion-matrix)
      - [Model C](#model-c)
        - [Classification Report for Deeper NN Model](#classification-report-for-deeper-nn-model)
        - [Confusion Matrix for Deeper NN Model](#confusion-matrix-for-deeper-nn-model)
    - [Convolutional Neural Network](#convolutional-neural-network)
      - [Architecture 1 : 1D Convolutional Neural Network](#architecture-1--1d-convolutional-neural-network)
      - [Architecture 2 : Regularized 1D Convolutional Neural Network](#architecture-2--regularized-1d-convolutional-neural-network)
      - [Architecture 3 : Deeper 1D Convolutional Neural Network](#architecture-3--deeper-1d-convolutional-neural-network)
    - [The Results](#the-results)
      - [Model A : 1D Convolutional Neural Network](#model-a--1d-convolutional-neural-network)
        - [Classification Report For CNN](#classification-report-for-cnn)
        - [Confusion Matrix For CNN](#confusion-matrix-for-cnn)
      - [Model C : Deeper 1D Convolutional Neural Network](#model-c--deeper-1d-convolutional-neural-network)
        - [Classification Report For Deeper CNN](#classification-report-for-deeper-cnn)
        - [Confusion Matrix For Deeper CNN](#confusion-matrix-for-deeper-cnn)
    - [Conclusion](#conclusion)
  - [The Segmentation Model](#the-segmentation-model)

## The Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/kinguistics/heartbeat-sounds).

## Data Exploratory analysis

Set A:

```python
	dataset	fname	label	sublabel
0	a	set_a/artifact__201012172012.wav	artifact	NaN
1	a	set_a/artifact__201105040918.wav	artifact	NaN
2	a	set_a/artifact__201105041959.wav	artifact	NaN
3	a	set_a/artifact__201105051017.wav	artifact	NaN
4	a	set_a/artifact__201105060108.wav	artifact	NaN
...	...	...	...	...
170	a	set_a/__201108222234.wav	NaN	NaN
171	a	set_a/__201108222241.wav	NaN	NaN
172	a	set_a/__201108222244.wav	NaN	NaN
173	a	set_a/__201108222247.wav	NaN	NaN
174	a	set_a/__201108222254.wav	NaN	NaN
175 rows × 4 columns
```
```python
setA.info()
```
```python
RangeIndex: 176 entries, 0 to 175
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   dataset   176 non-null    object 
 1   fname     176 non-null    object 
 2   label     124 non-null    object 
 3   sublabel  0 non-null      float64
dtypes: float64(1), object(3)
memory usage: 5.6+ KB
```
```python
print(setA["label"].value_counts())
```
artifact    40

murmur      34

normal      31

extrahls    19
```python
print(setA["label"].value_counts().sum())
```
124
```python
print(setA["label"].isnull().sum())
```
We should remove all null label entries from the dataframe
```python
setA = setA.dropna(subset=['label'])
```
We should also remove the artifact label entries since they are anomalies
```python
setA = setA[setA.label != 'artifact']
```
Now lets look at the altered value counts with the 3 classes
```python
print(setA["label"].value_counts())
```
murmur      34

normal      31

extrahls    19



Set B:

```python
	dataset	fname	label	sublabel
0	b	set_b/Btraining_extrastole_127_1306764300147_C...	extrastole	NaN
1	b	set_b/Btraining_extrastole_128_1306344005749_A...	extrastole	NaN
2	b	set_b/Btraining_extrastole_130_1306347376079_D...	extrastole	NaN
3	b	set_b/Btraining_extrastole_134_1306428161797_C...	extrastole	NaN
4	b	set_b/Btraining_extrastole_138_1306762146980_B...	extrastole	NaN
...	...	...	...	...
650	b	set_b/Btraining_normal_Btraining_noisynormal_2...	normal	noisynormal
651	b	set_b/Btraining_normal_Btraining_noisynormal_2...	normal	noisynormal
652	b	set_b/Btraining_normal_Btraining_noisynormal_2...	normal	noisynormal
653	b	set_b/Btraining_normal_Btraining_noisynormal_2...	normal	noisynormal
654	b	set_b/Btraining_normal_Btraining_noisynormal_2...	normal	noisynormal
655 rows × 4 columns
```
```python
setB.info()
```
```python
RangeIndex: 656 entries, 0 to 655
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   dataset   656 non-null    object
 1   fname     656 non-null    object
 2   label     461 non-null    object
 3   sublabel  149 non-null    object
dtypes: object(4)
memory usage: 20.6+ KB
```
```python
setB.describe()
```
```python
	dataset	fname	label	sublabel
count	656	656	461	149
unique	1	656	3	2
top	b	set_b/Btraining_extrastole_127_1306764300147_C...	normal	noisynormal
freq	656	1	320	120
```
We will do the same here as with set A and drop all entries with null values
This will give us 3 classes in set B
```python
setB = setB.dropna(subset=['label'])
```
```python
print(setB["label"].value_counts())
```
normal        320

murmur         95

extrastole     46

We will need all 4 categories together to be able to classify, so will join both sets A and B

Combining Sets A and B:
```python
setAB = [setA,setB]
ABdf = pd.concat(setAB)
print(ABdf.head(-1))
```
```python
dataset                                              fname     label  
40        a                   set_a/extrahls__201101070953.wav  extrahls   
41        a                   set_a/extrahls__201101091153.wav  extrahls   
42        a                   set_a/extrahls__201101152255.wav  extrahls   
43        a                   set_a/extrahls__201101160804.wav  extrahls   
44        a                   set_a/extrahls__201101160808.wav  extrahls   
..      ...                                                ...       ...   
650       b  set_b/Btraining_normal_Btraining_noisynormal_2...    normal   
651       b  set_b/Btraining_normal_Btraining_noisynormal_2...    normal   
652       b  set_b/Btraining_normal_Btraining_noisynormal_2...    normal   
653       b  set_b/Btraining_normal_Btraining_noisynormal_2...    normal   
654       b  set_b/Btraining_normal_Btraining_noisynormal_2...    normal         
[544 rows x 4 columns]
```
The distribution of each class:
normal        351

murmur        129

extrastole     46

extrahls       19

![AB-dist-before]

The data is very unbalanced, so we will upsample extrahls and extrastole and downsample normal

![AB-dist-after]

Set A timing:

```python
	fname	cycle	sound	location
0	set_a/normal__201102081321.wav	1	S1	10021
1	set_a/normal__201102081321.wav	1	S2	20759
2	set_a/normal__201102081321.wav	2	S1	35075
3	set_a/normal__201102081321.wav	2	S2	47244
4	set_a/normal__201102081321.wav	3	S1	62992
...	...	...	...	...
384	set_a/normal__201108011118.wav	10	S1	272527
385	set_a/normal__201108011118.wav	10	S2	284673
386	set_a/normal__201108011118.wav	11	S1	300863
387	set_a/normal__201108011118.wav	11	S2	314279
388	set_a/normal__201108011118.wav	12	S1	330980
389 rows × 4 columns
```
```python
setAtiming.info()
```
```python
RangeIndex: 390 entries, 0 to 389
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   fname     390 non-null    object
 1   cycle     390 non-null    int64 
 2   sound     390 non-null    object
 3   location  390 non-null    int64 
dtypes: int64(2), object(2)
memory usage: 12.3+ KB
```
```python
setAtiming.describe()
```
```python
	cycle	location
count	390.000000	390.000000
mean	5.733333	164639.984615
std	3.732807	99310.875752
min	1.000000	2583.000000
25%	3.000000	82313.000000
50%	5.000000	155624.500000
75%	8.000000	239709.750000
max	19.000000	390873.000000
```
Check to see if the lub and dub are the same amount
```python
print(setAtiming['sound'].value_counts())
```
S1    195

S2    195

Plot the cycles against their locations
```python
g= sns.histplot(data=setAtiming, x="cycle", y="location", cbar=True)
```
![cycles-locations]

Analyzing the Audios:

There are naming mistakes in the csv files so we renamed 'fname' to match the audio files.
We then created a new dataframe with the information needed.
```python
Audio	Label
0	drive/MyDrive/PR_assignment3/set_a/murmur__201...	murmur
1	drive/MyDrive/PR_assignment3/set_a/murmur__201...	murmur
2	drive/MyDrive/PR_assignment3/set_a/murmur__201...	murmur
3	drive/MyDrive/PR_assignment3/set_a/murmur__201...	murmur
4	drive/MyDrive/PR_assignment3/set_a/murmur__201...	murmur
...	...	...
510	drive/MyDrive/PR_assignment3/set_b/normal_nois...	normal
511	drive/MyDrive/PR_assignment3/set_b/normal__159...	normal
512	drive/MyDrive/PR_assignment3/set_b/normal_nois...	normal
513	drive/MyDrive/PR_assignment3/set_a/normal__201...	normal
514	drive/MyDrive/PR_assignment3/set_b/normal_nois...	normal
515 rows × 2 columns
```
1) Extrahls

![extrahls-wav]

![extrahls-wav-reduced-noise]

![extrahls-spectrum]

Spectrogram to represent the noise or sound intensity of audio data with respect to frequency and time

![extrahls-spectogram]

Feature Extraction from audio:

Visualize audio data focused on a particular point or mean (centroid)

![extrahls-centroids]

MFCCs small set of features that describe the overall shape of a spectral envelope

![extrahls-mfccs]

2) Murmur

![murmur-wav]

![murmur-wav-reduced-noise]

![murmur-spectrum]

Spectrogram to represent the noise or sound intensity of audio data with respect to frequency and time

![murmur-spectogram]

Feature Extraction from audio:

Visualize audio data focused on a particular point or mean (centroid)

![murmur-centroids]

MFCCs small set of features that describe the overall shape of a spectral envelope

![murmur-mfccs]

3) Normal

![normal-wav]

![normal-wav-reduced-noise]

![normal-spectrum]

Spectrogram to represent the noise or sound intensity of audio data with respect to frequency and time

![normal-spectogram]

Feature Extraction from audio:

Visualize audio data focused on a particular point or mean (centroid)

![normal-centroids]

MFCCs small set of features that describe the overall shape of a spectral envelope

![normal-mfccs]

4) Extrastole

![extrastole-wav]

![extrastole-wav-reduced-noise]

![extrastole-spectrum]

Spectrogram to represent the noise or sound intensity of audio data with respect to frequency and time

![extrastole-spectogram]

Feature Extraction from audio:

Visualize audio data focused on a particular point or mean (centroid)

![extrastole-centroids]

MFCCs small set of features that describe the overall shape of a spectral envelope

![extrastole-mfccs]

## Data Splitting
Split the data 70 train, 15 validation, 15 test.

For task A (size):

x test = 177

x test shape = (59, 3)

y test = 59

x train = 816

x train shape = (272, 3)

y train = 272

x val = 177

x validation shape = (59, 3)

y val = 59

For task B (size):

x test = 78

y test = 78

x train = 360

y train = 360

x val = 78

y val = 78

## The Classification Model

[![kaggle-badge]][classification-notebook]

### Data Preprocessing

The data is prepared for classification using the following steps:

1. The audio files are sampled at a constant rate of 22050 Hz.
2. The shorter audio files are padded with zeros to match the length of the longest audio file at 12 seconds.
3. The lables are one-hot encoded.

```python
labels = {
    'murmur'     : np.array([1,0,0,0]),
    'normal'     : np.array([0,1,0,0]),
    'extrahls'   : np.array([0,0,1,0]),
    'extrastole' : np.array([0,0,0,1]),
}
```

### Hyperparameters

| Hyperparameter          | Value                     |
| ----------------------- | ------------------------- |
| Learning Rate Schedule  | Step Decay                |
| Learning Rate Factor    | 2e-5                      |
| Learning Rate Patience  | 35                        |
| Activation Function     | ReLU , Softmax            |
| Optimizer               | Adam                      |
| Loss Function           | Categorical Cross Entropy |
| Epochs                  | 500                       |
| Early Stopping          | True                      |
| Early Stopping Patience | 50                        |

### Feed Forward Fully Connected Neural Network

We try different architectures for the fully connected neural network.

#### Architecture 1

```text

Model: "ClassifierA"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
_________________________________________________________________
flatten_21 (Flatten)         (None, 40)                0
_________________________________________________________________
dense_57 (Dense)             (None, 2048)              83968
_________________________________________________________________
dense_58 (Dense)             (None, 512)               1049088
_________________________________________________________________
dense_59 (Dense)             (None, 4)                 2052
_________________________________________________________________
Total params: 1,135,108
Trainable params: 1,135,108
Non-trainable params: 0
_________________________________________________________________

```

| Information         | Value |
| ------------------- | ----- |
| Number of epochs    | 171   |
| Training Accuracy   | 0.93  |
| Training Loss       | 0.24  |
| Validation Accuracy | 0.78  |
| Validation Loss     | 0.72  |

![loss][loss-nn]

![accuracy][accuracy-nn]

#### Architecture 2

We add regularization to the model to prevent overfitting.

```text
Model: "ClassifierB"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten_22 (Flatten)         (None, 40)                0
_________________________________________________________________
dense_60 (Dense)             (None, 2048)              83968
_________________________________________________________________
dense_61 (Dense)             (None, 512)               1049088
_________________________________________________________________
dropout_13 (Dropout)         (None, 512)               0
_________________________________________________________________
dense_62 (Dense)             (None, 4)                 2052
=================================================================
Total params: 1,135,108
Trainable params: 1,135,108
Non-trainable params: 0
_________________________________________________________________
```

Number of epochs : 308

Training Accuracy : 0.89

Training Loss : 0.30

Validation Accuracy : 0.74

Validation Loss : 0.61

![loss][loss-nn-reg]

![accuracy][accuracy-nn-reg]

#### Architecture 3 : Deeper

```text
Model: "ClassifierC"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten_29 (Flatten)         (None, 40)                0
_________________________________________________________________
dense_85 (Dense)             (None, 2048)              83968
_________________________________________________________________
dense_86 (Dense)             (None, 1024)              2098176
_________________________________________________________________
dense_87 (Dense)             (None, 64)                65600
_________________________________________________________________
dense_88 (Dense)             (None, 64)                4160
_________________________________________________________________
dense_89 (Dense)             (None, 4)                 260
=================================================================
Total params: 2,252,164
Trainable params: 2,252,164
Non-trainable params: 0
_________________________________________________________________
```

Number of epochs : 169

Training Accuracy : 0.86

Training Loss : 0.32

Validation Accuracy : 0.76

Validation Loss : 0.63

![loss][loss-nn-deep]

![accuracy][accuracy-nn-deep]

### Test Results For Feed Forward Fully Connected Neural Network

The best model is the first model.

| Model | Accuracy | Loss | AUC  |
| ----- | -------- | ---- | ---- |
| C     | 0.76     | 0.63 | 0.97 |
| A     | 0.83     | 0.71 | 0.92 |

#### Model A

##### Classification Report

```text
              precision    recall  f1-score   support

      murmur       0.86      0.71      0.77        17
      normal       0.75      0.63      0.69        19
    extrahls       0.95      1.00      0.98        20
  extrastole       0.78      0.95      0.86        22

    accuracy                           0.83        78
   macro avg       0.83      0.82      0.82        78
weighted avg       0.83      0.83      0.83        78
```

##### Confusion Matrix

![confusion matrix][confusion-nn]

#### Model C

##### Classification Report for Deeper NN Model

```text
              precision    recall  f1-score   support

      murmur       0.65      0.76      0.70        17
      normal       0.80      0.42      0.55        19
    extrahls       0.95      1.00      0.98        20
  extrastole       0.78      0.95      0.86        22

    accuracy                           0.79        78
   macro avg       0.80      0.79      0.77        78
weighted avg       0.80      0.79      0.78        78
```

##### Confusion Matrix for Deeper NN Model

![confusion matrix][confusion-nn-deep]

### Convolutional Neural Network

We try different architectures for the convolutional neural network.

#### Architecture 1 : 1D Convolutional Neural Network

```text
Model: "sequential_9"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d_38 (Conv1D)           (None, 40, 64)            256
_________________________________________________________________
conv1d_39 (Conv1D)           (None, 40, 64)            12352
_________________________________________________________________
max_pooling1d_19 (MaxPooling (None, 20, 64)            0
_________________________________________________________________
conv1d_40 (Conv1D)           (None, 20, 32)            6176
_________________________________________________________________
conv1d_41 (Conv1D)           (None, 20, 32)            3104
_________________________________________________________________
max_pooling1d_20 (MaxPooling (None, 10, 32)            0
_________________________________________________________________
flatten_23 (Flatten)         (None, 320)               0
_________________________________________________________________
dense_63 (Dense)             (None, 64)                20544
_________________________________________________________________
dense_64 (Dense)             (None, 4)                 260
=================================================================
Total params: 42,692
Trainable params: 42,692
Non-trainable params: 0
_________________________________________________________________
```

| Information         | Value  |
| ------------------- | ------ |
| Number of epochs    | 145    |
| Training Accuracy   | 0.9667 |
| Training Loss       | 0.1258 |
| Validation Accuracy | 0.8333 |
| Validation Loss     | 0.6647 |

![loss][loss-cnn]

![accuracy][accuracy-cnn]

#### Architecture 2 : Regularized 1D Convolutional Neural Network

We add batch normalization and dropout.

```text
Model: "sequential_10"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d_42 (Conv1D)           (None, 40, 64)            256
_________________________________________________________________
conv1d_43 (Conv1D)           (None, 40, 64)            12352
_________________________________________________________________
max_pooling1d_21 (MaxPooling (None, 20, 64)            0
_________________________________________________________________
batch_normalization_8 (Batch (None, 20, 64)            256
_________________________________________________________________
conv1d_44 (Conv1D)           (None, 20, 32)            6176
_________________________________________________________________
conv1d_45 (Conv1D)           (None, 20, 32)            3104
_________________________________________________________________
max_pooling1d_22 (MaxPooling (None, 10, 32)            0
_________________________________________________________________
batch_normalization_9 (Batch (None, 10, 32)            128
_________________________________________________________________
flatten_24 (Flatten)         (None, 320)               0
_________________________________________________________________
dropout_14 (Dropout)         (None, 320)               0
_________________________________________________________________
dense_65 (Dense)             (None, 64)                20544
_________________________________________________________________
dropout_15 (Dropout)         (None, 64)                0
_________________________________________________________________
dense_66 (Dense)             (None, 4)                 260
=================================================================
Total params: 43,076
Trainable params: 42,884
Non-trainable params: 192
_________________________________________________________________
```

> Note: The model is trained for 500 epochs because we do not use early stopping.

| Information         | Value  |
| ------------------- | ------ |
| Number of epochs    | 500    |
| Training Accuracy   | 0.9972 |
| Training Loss       | 0.02   |
| Validation Accuracy | 0.7821 |
| Validation Loss     | 0.7693 |

![loss][loss-cnn-reg]

![accuracy][accuracy-cnn-reg]

#### Architecture 3 : Deeper 1D Convolutional Neural Network

We add more layers to the model.

```text
Model: "sequential_11"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d_46 (Conv1D)           (None, 40, 64)            256
_________________________________________________________________
conv1d_47 (Conv1D)           (None, 40, 64)            12352
_________________________________________________________________
max_pooling1d_23 (MaxPooling (None, 20, 64)            0
_________________________________________________________________
conv1d_48 (Conv1D)           (None, 20, 32)            6176
_________________________________________________________________
conv1d_49 (Conv1D)           (None, 20, 32)            3104
_________________________________________________________________
max_pooling1d_24 (MaxPooling (None, 10, 32)            0
_________________________________________________________________
conv1d_50 (Conv1D)           (None, 10, 16)            1552
_________________________________________________________________
conv1d_51 (Conv1D)           (None, 10, 16)            784
_________________________________________________________________
max_pooling1d_25 (MaxPooling (None, 5, 16)             0
_________________________________________________________________
flatten_25 (Flatten)         (None, 80)                0
_________________________________________________________________
dense_67 (Dense)             (None, 64)                5184
_________________________________________________________________
dense_68 (Dense)             (None, 64)                4160
_________________________________________________________________
dense_69 (Dense)             (None, 4)                 260
=================================================================
Total params: 33,828
Trainable params: 33,828
Non-trainable params: 0
_________________________________________________________________
```

| Information         | Value  |
| ------------------- | ------ |
| Number of epochs    | 144    |
| Training Accuracy   | 0.9972 |
| Training Loss       | 0.0565 |
| Validation Accuracy | 0.8077 |
| Validation Loss     | 0.7947 |

![loss][loss-cnn-deep]

![accuracy][accuracy-cnn-deep]

### The Results

The best model is the first model.

| Model | Accuracy | Loss   | AUC   |
| ----- | -------- | ------ | ----- |
| A     | 0.7436   | 0.5520 | 0.946 |
| C     | 0.73     | 0.79   | 0.92  |

#### Model A : 1D Convolutional Neural Network

##### Classification Report For CNN

```text
              precision    recall  f1-score   support

      murmur       0.65      0.65      0.65        17
      normal       0.53      0.42      0.47        19
    extrahls       0.95      1.00      0.98        20
  extrastole       0.76      0.86      0.81        22

    accuracy                           0.74        78
   macro avg       0.72      0.73      0.73        78
weighted avg       0.73      0.74      0.73        78
```

##### Confusion Matrix For CNN

![confusion matrix][confusion-cnn]

#### Model C : Deeper 1D Convolutional Neural Network

##### Classification Report For Deeper CNN

```text
              precision    recall  f1-score   support

      murmur       0.65      0.65      0.65        17
      normal       0.50      0.53      0.51        19
    extrahls       0.95      1.00      0.98        20
  extrastole       0.80      0.73      0.76        22

    accuracy                           0.73        78
   macro avg       0.72      0.73      0.72        78
weighted avg       0.73      0.73      0.73        78
```

##### Confusion Matrix For Deeper CNN

![confusion matrix][confusion-cnn-deep]

### Regression model
#### feed forward network
model architechture:
```python
def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(2048, input_dim=dim, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    # check to see if the regression node should be added
    if regress:
      model.add(Dense(1, activation="linear"))
    # return our model
    return model
```
adam optmizer:
```python
import tensorflow as tf
values = np.arange(0.000001,0.0003,0.00002)[::-1]
# values = np.array([0.00003,0.00005,0.00007,0.00009,0.0001,0.0003])[::-1]
boundaries = np.arange(10, 600,35)[:values.shape[0]-1]


scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    list(boundaries), list(values))

lrscheduler = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)
```
metrics:
```python
import tensorflow_addons as tfa
metric = tfa.metrics.r_square.RSquare()
model=create_mlp(20,regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mse", optimizer=opt,metrics=metric)
```
training loss, rsquareloss = 0.06839843094348907, -0.07385599613189697

Mean absolute error = 0.22

Mean squared error = 0.07

Median absolute error = 0.22

Explain variance score = 0.0

R2 score = -0.03

mean -12.865519

std 160.42049

graphs:
![ff_tl][ff_tl]
![ff_r2s][ff_r2s]
#### CNN
model architechture:
```python
from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
model = Sequential()
model.add(Conv1D(256, 2, activation="relu", input_shape=(20, 1)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1,activation="linear"))
metric = tfa.metrics.r_square.RSquare()
model.compile(loss="mse", optimizer="adam",metrics=metric)
```
same learning rate and meterics as feed forward network

training loss,rsquare loss=[0.06612320989370346, 0.0006309747695922852]

Mean absolute error = 0.22

Mean squared error = 0.07

Median absolute error = 0.22

Explain variance score = 0.0

R2 score = -0.03


graphs:
![cnn_tl][cnn_tl]
![cnn_r2s][cnn_r2s]
### Conclusion

- CNN models take much less time to train than feed forward networks.
- The accuracy of the CNN models is not as good as the feed forward networks.
- Regularization by adding dropout does not always prevent overfitting.

## The Segmentation Model

<!-- References -->

<!-- Links -->
[github-badge]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white

[repo]: https://github.com/moharamfatema/heartbeat-sounds

[kaggle-badge]: https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white

[classification-notebook]: https://www.kaggle.com/code/fatemamoharam/heartbeat-sounds-classification/notebook

<!-- Images -->

[cycles-locations]: img/set_a_timing_location_cycles.PNG

[AB-dist-before]: img/distribution_AB_before.PNG

[AB-dist-after]: img/distribution_AB_after.PNG

[extrahls-wav]: img/extrahls_wav.PNG

[extrahls-wav-reduced-noise]: img/extrahls_wav_reduced_noise.PNG

[extrahls-spectrum]: img/extrahls_spectrum.PNG

[extrahls-spectogram]: img/extrahls_spectogram.PNG

[extrahls-centroids]: img/extrahls_centroids.PNG

[extrahls-mfccs]: img/extrahls_mfccs.PNG

[murmur-wav]: img/mumur_wav.PNG
[murmur-wav-reduced-noise]: img/murmur_wav_reduced_noise.PNG

[murmur-spectrum]: img/murmur_spectrum.PNG

[murmur-spectogram]: img/murmur_spectogram.PNG

[murmur-centroids]: img/murmur_centroids.PNG

[murmur-mfccs]: img/murmur_mfccs.PNG

[normal-wav]: img/normal_wav.PNG

[normal-wav-reduced-noise]: img/normal_wav_reduced_noise.PNG

[normal-spectrum]: img/normal_spectrum.PNG

[normal-spectogram]: img/normal_spectogram.PNG

[normal-centroids]: img/normal_centroids.PNG

[normal-mfccs]: img/normal_mfccs.PNG

[extrastole-wav]: img/extrastole_wav.PNG

[extrastole-wav-reduced-noise]: img/extrastole_wav_reduced_noise.PNG

[extrastole-spectrum]: img/extrastole_spectrum.PNG

[extrastole-spectogram]: img/extrastole_spectogram.PNG

[extrastole-centroids]: img/extrastole_centroids.PNG

[extrastole-mfccs]: img/extrastole_mfccs.PNG

[loss-nn]: img/lossnn.png

[accuracy-nn]: img/accnn.png

[loss-nn-reg]: img/lossnnreg.png

[accuracy-nn-reg]: img/accnnreg.png

[loss-nn-deep]: img/lossnndeep.png

[accuracy-nn-deep]: img/accnndeep.png

[confusion-nn-deep]: img/confnndeep.png

[confusion-nn]: img/confnn.png

[loss-cnn]: img/losscnn.png

[accuracy-cnn]: img/acccnn.png

[loss-cnn-reg]: img/losscnnreg.png

[accuracy-cnn-reg]: img/acccnnreg.png

[loss-cnn-deep]: img/losscnndeep.png

[accuracy-cnn-deep]: img/acccnndeep.png

[confusion-cnn-deep]: img/confcnndeep.png

[confusion-cnn]: img/confcnn.png
[ff_tl]: img/ff_tl.PNG
[ff_r2s]: img/ff_r2s.PNG
[cnn_tl]: img/cnn_tl.PNG
[cnn_r2s]: img/cnn_r2s.PNG

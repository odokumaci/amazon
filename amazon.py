
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt
from keras.models import load_model

# Set seed for reproducibity
np.random.seed(2017)

# Read in image tags and labels
df_train_all = pd.read_csv('../input/train_v2.csv')
split = 36431 #10% val split
df_train, df_valid = df_train_all[:split], df_train_all[split:]

labels = ['cloudy',
 'blow_down',
 'primary',
 'bare_ground',
 'habitation',
 'selective_logging',
 'clear',
 'cultivation',
 'slash_burn',
 'agriculture',
 'road',
 'blooming',
 'water',
 'haze',
 'artisinal_mine',
 'conventional_mine',
 'partly_cloudy']

label_map = {'agriculture': 9,
 'artisinal_mine': 14,
 'bare_ground': 3,
 'blooming': 11,
 'blow_down': 1,
 'clear': 6,
 'cloudy': 0,
 'conventional_mine': 15,
 'cultivation': 7,
 'habitation': 4,
 'haze': 13,
 'partly_cloudy': 16,
 'primary': 2,
 'road': 10,
 'selective_logging': 5,
 'slash_burn': 8,
 'water': 12}

y_valid = []

for f, tags in tqdm(df_valid.values, miniters=1000):
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    y_valid.append(targets)

y_valid = np.array(y_valid, np.uint8)

# Model
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

# create the base pre-trained model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(128,128,3))

x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)

predictions = Dense(17, activation='sigmoid')(x)

# final model
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

#train the model
batch_size = 128

train_datagen = ImageDataGenerator(
#    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


# generators for memory efficient batch processing
def train_generator():
        while True:
            for start in range(0, len(df_train), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_train))
                df_train_batch = df_train[start:end]
                for f, tags in df_train_batch.values:
                    img = cv2.imread('../input/train/{}.png'.format(f))
                    img = train_datagen.random_transform(img)
                    targets = np.zeros(17)
                    for t in tags.split(' '):
                        targets[label_map[t]] = 1
                    x_batch.append(img)
                    y_batch.append(targets)
                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch

def valid_generator():
        while True:
            for start in range(0, len(df_valid), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_valid))
                df_valid_batch = df_valid[start:end]
                for f, tags in df_valid_batch.values:
                    img = cv2.imread('../input/train/{}.png'.format(f))
                    targets = np.zeros(17)
                    for t in tags.split(' '):
                        targets[label_map[t]] = 1
                    x_batch.append(img)
                    y_batch.append(targets)
                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch

def pred_generator():
        while True:
            for start in range(0, len(df_valid), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_valid))
                df_valid_batch = df_valid[start:end]
                for f, tags in df_valid_batch.values:
                    img = cv2.imread('../input/train/{}.png'.format(f))
                    targets = np.zeros(17)
                    for t in tags.split(' '):
                        targets[label_map[t]] = 1
                    x_batch.append(img)
                    y_batch.append(targets)
                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch


# set learning rate schedule
from keras.callbacks import LearningRateScheduler

def lrate_epoch(epoch):
   epochs_arr = [0, 10, 20, 30]
   learn_rates = [1e-4, 1e-5, 1e-6]
   #epochs_arr = [0, 5]
   #learn_rates = [0.0001]
   lrate = learn_rates[0]
   if (epoch > epochs_arr[len(epochs_arr)-1]):
           lrate = learn_rates[len(epochs_arr)-2]
   for i in range(len(epochs_arr)-1):
       if (epoch > epochs_arr[i] and epoch <= epochs_arr[i+1]):
           lrate = learn_rates[i]
   return lrate

lrateschedule = LearningRateScheduler(lrate_epoch)

# set optimizer and compile model
from keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# set callbacks
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='amazon_best.h5', verbose=1, save_best_only=True)

from keras.callbacks import CSVLogger
csv_logger = CSVLogger('amazon.csv', append = True)

callbacks = [checkpointer, csv_logger, lrateschedule]


#Finally fit the model
model.fit_generator(generator=train_generator(),
                        steps_per_epoch=(len(df_train) // batch_size) + 1,
                        epochs=30,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_generator(),
                        validation_steps=(len(df_valid) // batch_size) + 1)

model.save("amazon_final.h5")

#Reload the model weights and evaluate it on validation set
model.load_weights("amazon_best.h5")

score = model.evaluate_generator(generator=valid_generator(), steps=(len(df_valid) // batch_size) + 1 )

from sklearn.metrics import fbeta_score

p_valid = model.predict_generator(generator=pred_generator(), steps=(len(df_valid) // batch_size) + 1 )


#optimize thresholds to maximize f2_score
def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    #credits https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*17
  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= resolution
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x

best_threshold = optimise_f2_thresholds(y_valid, p_valid)

print(fbeta_score(y_valid, np.array(p_valid) > best_threshold, beta=2, average='samples'))

#Submission

df_test = pd.read_csv('../input/sample_submission_v2.csv')
df_test.head()


#test time augmentation

def test_pred_generator():
        while True:
            for start in range(0, len(df_test), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_test))
                df_test_batch = df_test[start:end]
                for f, tags in df_test_batch.values:
                    img = cv2.imread('../input/test/{}.png'.format(f))
                    img = train_datagen.random_transform(img) #tta
                    targets = np.zeros(17)
                    for t in tags.split(' '):
                        targets[label_map[t]] = 1
                    x_batch.append(img)
                    y_batch.append(targets)
                x_batch = np.array(x_batch, np.float32) / 255.
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch

TTA_steps = 3

y_test = []
for k in range(0, TTA_steps):
    print(k)
    p_test = model.predict_generator(generator=test_pred_generator(), steps=(len(df_test) // batch_size) + 1, verbose=1 )
    y_test.append(p_test)
    k += 1

result = np.array(y_test[0])

for i in range(1, TTA_steps):
    result += np.array(y_test[i])
result /= TTA_steps

result = pd.DataFrame(result, columns=labels)

preds = []

for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > best_threshold, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test['tags'] = preds
df_test.to_csv('submission_amazon.csv', index=False)

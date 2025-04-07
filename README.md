```markdown
# Using Convolutions with Complex Images

This project demonstrates how to build and train a convolutional neural network (CNN) to distinguish between horses and humans in images. Unlike simpler datasets where the subject is centered (e.g., Fashion MNIST), this lab tackles a more challenging problem where the subject can appear anywhere in the image.

The network is designed to:
- Download and extract the dataset.
- Automatically label images by reading from subdirectories.
- Visualize example images from both classes.
- Build a CNN from scratch with convolutional and pooling layers.
- Preprocess the image data using an ImageDataGenerator.
- Train the network using binary crossentropy loss and the RMSprop optimizer.
- Evaluate the model and run predictions on new images.
- Visualize intermediate representations to gain insight into the CNN's feature extraction process.

---

## Table of Contents

- [License](#license)
- [Overview](#overview)
- [Dataset Download and Extraction](#dataset-download-and-extraction)
- [Data Exploration and Visualization](#data-exploration-and-visualization)
- [Building the Model](#building-the-model)
- [Data Preprocessing](#data-preprocessing)
- [Training the Model](#training-the-model)
- [Running Predictions](#running-predictions)
- [Visualizing Intermediate Representations](#visualizing-intermediate-representations)
- [Clean Up](#clean-up)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

---

## License

Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at:

[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and limitations under the License.

---

## Overview

In this lab, we train a CNN to classify images as either "horse" or "human." The network must learn to detect features from images where the subject is not necessarily centered. We use image generators to automatically label the data based on directory structure, and we apply convolutional layers to extract spatial features.

---

## Dataset Download and Extraction

The dataset is downloaded using `wget` and extracted using Python's `zipfile` module. Two zip files are used: one for training and one for validation.

Example commands to download:
```bash
!wget --no-check-certificate \
    https://storage.googleapis.com/learning-datasets/horse-or-human.zip \
    -O /tmp/horse-or-human.zip

!wget --no-check-certificate \
    https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip \
    -O /tmp/validation-horse-or-human.zip
```

Extraction is done with:
```python
import os
import zipfile

local_zip = '/tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/horse-or-human')

local_zip = '/tmp/validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation-horse-or-human')
zip_ref.close()
```

The extracted directories contain subdirectories for `horses` and `humans`, which will be used for automatic labeling.

---

## Data Exploration and Visualization

The code sets up directory paths and lists file names to inspect the dataset:
```python
# Define directories for training images
train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
train_human_dir = os.path.join('/tmp/horse-or-human/humans')

# Define directories for validation images
validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')
validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')

# List some file names
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

# Print total counts
print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))
```

Visualization using Matplotlib displays batches of images:
```python
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix + next_human_pix):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()
```

---

## Building the Model

The model is built using TensorFlow 2.x. It consists of:
- Convolutional layers with ReLU activation.
- MaxPooling layers to reduce spatial dimensions.
- A Flatten layer to convert the 2D feature maps to a 1D vector.
- Dense layers with a final sigmoid activation for binary classification.

```python
try:
    %tensorflow_version 2.x
except Exception:
    pass

import tensorflow as tf
print(tf.__version__)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
```

---

## Data Preprocessing

Image data is preprocessed using the `ImageDataGenerator`:
- Images are rescaled to normalize pixel values to the range [0, 1].
- Training and validation generators are created from the respective directories.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
    '/tmp/horse-or-human/',
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(
    '/tmp/validation-horse-or-human/',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)
```

---

## Training the Model

The model is compiled using the RMSprop optimizer with a learning rate of 0.001 and binary crossentropy loss. Training is performed for 15 epochs.

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

history = model.fit(
      train_generator,
      validation_data=validation_generator,
      epochs=15,
      steps_per_epoch=8,
      validation_steps=8,
      verbose=1
)
```

---

## Running Predictions

After training, you can test the model on new images. The following code uploads images, preprocesses them, and prints whether each image is classified as a horse or a human.

```python
import numpy as np
from google.colab import files
from tensorflow.keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
    path = '/content/' + fn
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(fn + " is a human")
    else:
        print(fn + " is a horse")
```

---

## Visualizing Intermediate Representations

To understand what the CNN has learned, you can visualize the intermediate feature maps. This code extracts outputs from each layer (after the first) and displays them.

```python
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)

img = load_img(img_path, target_size=(300, 300))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
x /= 255

successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers]

import matplotlib.pyplot as plt
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            x = feature_map[0, :, :, i]
            x -= x.mean()
            if x.all() > 0:
                x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = x
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
```

---

## Clean Up

Before running additional experiments, free memory resources by terminating the kernel with:

```python
import os, signal
os.kill(os.getpid(), signal.SIGKILL)
```

---

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/horse-or-human-cnn.git
   cd horse-or-human-cnn
   ```

2. **Install Dependencies:**

   Ensure you have TensorFlow 2.x and other required libraries installed:
   ```bash
   pip install tensorflow matplotlib
   ```

3. **Run the Notebook or Script:**

   - For a Jupyter Notebook, open the provided `.ipynb` file and run the cells sequentially.
   - For a script, execute:
     ```bash
     python your_script.py
     ```

4. **Experiment:**

   - Test the model with new images.
   - Visualize intermediate layers to understand feature extraction.
   - Adjust parameters such as epochs, learning rate, and network architecture to experiment with model performance.

---

## Acknowledgments

- **TensorFlow and Keras:** For providing powerful tools for building and training CNNs.
- **Google Colab:** For an easy-to-use platform to run the code.
- **Learning Datasets:** For hosting the horse-or-human dataset.
- **Apache License:** This project is distributed under the Apache License, Version 2.0.


# Planet: Understanding the Amazon from Space Challenge in Kaggle

Classification of ~40,000 images from Amazon into 17 different classes. The python script fine tunes a VGG19 neural network that is pretrained on ImageNet set. The data is augmented with random rotations, flips, shifts and zooms using Keras' own image generator. Traiining is done via batches to overcome memory limitations. This is achived through Keras generators.

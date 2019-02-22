# Planet: Understanding the Amazon from Space Challenge in Kaggle

Classification of ~40,000 images from Amazon into 17 different classes. The python script fine tunes a VGG19 neural network that is pretrained on ImageNet set. The data is augmented with random rotations, flips, shifts and zooms using Keras' own image generator. Training is done via batches to overcome memory limitations. This is achieved through Keras generators.
The challenge is evaluated based on the F2 score. F2 score is a blend of precision and recall with more weight towards recall in this competition. In the script, probability thresholds are optimized to maximize the F2 score. Finally test time augmentation is done on test samples before result submission. 

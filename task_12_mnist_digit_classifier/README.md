# Task 12: Handwritten digit classifier (CNN)

A CNN in PyTorch trained on MNIST.

## How to run

```
pip install torch torchvision numpy matplotlib seaborn scikit-learn jupyter
cd task_12_mnist_digit_classifier
jupyter notebook mnist_cnn.ipynb
```

MNIST is downloaded automatically into a `data` folder the first time. Training takes about 5 minutes on a CPU (6 epochs).

## Model

2 conv layers (32 and 64 filters, 3x3), max pooling, dropout, then 2 fully connected layers (128 and 10). Adam optimizer, cross entropy loss, batch size 128.

## Result

Test accuracy: **99.18%** (82 out of 10,000 test images wrong)

Validation accuracy after each epoch: 98.00, 98.58, 98.52, 98.86, 98.92, 99.04

The notebook also has the training curves, confusion matrix and some of the wrong predictions (they are mostly messy handwriting). The trained weights are saved in `model/mnist_cnn.pt`.

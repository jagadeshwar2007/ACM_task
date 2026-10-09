# Task 12 — Handwritten Digit Classifier (CNN, PyTorch)

A **Convolutional Neural Network** trained on **MNIST** (60,000 train / 10,000 test images of handwritten digits).

## Run it

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn jupyter
cd task_12_mnist_digit_classifier
jupyter notebook mnist_cnn.ipynb      # run all cells; MNIST downloads automatically into ./data
```

Trains in about **5 minutes on a 2-core CPU** (6 epochs, ~52 s each); no GPU needed. The notebook is already executed, so the outputs are visible on GitHub.

## Result

> ## Test accuracy: **99.18%** (82 of 10,000 test images misclassified)

| Epoch | Train loss | Validation loss | Validation accuracy |
|---|---|---|---|
| 1 | 0.2413 | 0.0666 | 98.00% |
| 2 | 0.0907 | 0.0507 | 98.58% |
| 3 | 0.0668 | 0.0515 | 98.52% |
| 4 | 0.0429 | 0.0422 | 98.86% |
| 5 | 0.0356 | 0.0409 | 98.92% |
| 6 | 0.0319 | 0.0411 | 99.04% |

The test set was only used once, at the end; 5,000 training images were held out as a validation set to monitor training.

## Model

`Conv(1→32, 3×3) → ReLU → Conv(32→64, 3×3) → ReLU → MaxPool(2) → Dropout(0.25) → Flatten → Linear(9216→128) → ReLU → Dropout(0.5) → Linear(128→10)`

About 1.2M parameters. Adam optimizer (lr 1e-3, decayed ×0.3 after epoch 3), cross-entropy loss, batch size 128, inputs normalised with MNIST's mean/std, fixed random seed.

## Outputs (`outputs/`)
`sample_digits.png`, `training_curves.png`, `confusion_matrix.png`, `misclassified.png`, `test_accuracy.txt`

The saved weights are in `model/mnist_cnn.pt` (load with `CNN().load_state_dict(torch.load(...))`; the class is defined in the notebook).
The remaining errors are mostly messy or genuinely ambiguous handwriting (see `misclassified.png`).

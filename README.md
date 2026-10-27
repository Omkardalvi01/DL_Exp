# Deep Learning Laboratory

Six independent notebooks implementing the six `DL Exp` PDFs from Downloads.

| Notebook | Experiment |
| --- | --- |
| [01_logic_gates.ipynb](01_logic_gates.ipynb) | AND, OR, NAND and XOR using NumPy and backpropagation |
| [02_mnist_models.ipynb](02_mnist_models.ipynb) | ANN, basic CNN and LeNet-5 style CNN comparison |
| [03_regularization.ipynb](03_regularization.ipynb) | Baseline CNN versus L2, dropout, batch normalization and augmentation |
| [04_optimization.ipynb](04_optimization.ipynb) | Seven optimizers, hyperparameter search, residual CNN, pruning, integer quantization and distillation |
| [05_rnn_text_classification.ipynb](05_rnn_text_classification.ipynb) | IMDB sentiment analysis using text vectorization and bidirectional LSTMs |
| [06_autoencoders.ipynb](06_autoencoders.ipynb) | Dense reconstruction, dense denoising and convolutional denoising autoencoders |

Use Python 3.11 or 3.12, or upload the notebooks to Google Colab. For local use:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open a notebook and run its cells from top to bottom. All training code is in the notebooks; they do not import project helper files. The first run needs internet access for MNIST and IMDB. Experiment 5 also accepts `data/IMDB-Dataset.csv` with `review` and `sentiment` columns. Training artifacts go into separate folders under `outputs/`.

Experiment 5 uses the Keras IMDB dataset from Google storage instead of downloading the archive from Stanford. Its word sequences are decoded to text for the vectorization and LSTM pipeline; original punctuation is not preserved. A local CSV or an extracted `data/aclImdb` / `data/imdb/**/aclImdb` directory takes priority. For offline Keras loading, copy the original `imdb.npz` and `imdb_word_index.json` files into `.keras/datasets/`.

Experiments 2–6 have `QUICK_RUN = False` in their configuration cell. Set it to `True` for a small demonstration. All Keras training runs use early stopping on validation loss and restore the best weights. Each call sets its patience; the epoch settings are upper limits. Experiments 2 and 3 allow up to 50 epochs by default. Experiments 5 and 6 allow up to 10 epochs, or 1 epoch in quick mode. The NumPy logic gates stop when training loss reaches 0.01 or stops improving, retaining the best weights.

Experiment 4 caps every training stage at 10 epochs, or 1 epoch in quick mode, with early stopping enabled. It includes genuine batch GD and stochastic GD; their batch sizes produce different numbers of updates per epoch. Its comparison reports actual epochs and update counts, since early stopping can end runs at different times. Its latency measurements use warmed-up, single-image inference; Python allocation measurements exclude native TensorFlow memory.

Experiment 6 implements both the exercise's single dense encoder/decoder and the convolutional model shown in the PDF. Experiment 5's “Neutral” output is an uncertainty band from a binary classifier, not a learned third sentiment class.

Metrics are calculated from the actual run. The notebooks do not copy the PDFs' reported accuracy values or contain fabricated training outputs.

Validation in this workspace: all notebook JSON and Python cells were checked, and the NumPy networks learned all four truth tables. Experiment 4 feature extraction passed on random and blank images. Experiment 5 passed loading, one training epoch, evaluation and prediction using small synthetic CSV and encoded-data fixtures; these checks do not measure IMDB accuracy. Full dataset training remains unverified, and sandbox DNS restrictions prevented checking the live IMDB download. The TensorFlow version range retains the bundled TFLite interpreter used in experiment 4.

API references: [Keras TextVectorization](https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/), [TensorFlow integer quantization](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/post_training_quantization.md).

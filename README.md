# Music Genre CNN
 This repository contains work from a project in my neural networks class at NC State. Our goal was to create a music genre classifier for the GTZAN dataset. To do this, we converted audio snippets to spectrograms and fed the magnitude plots of these spectrograms to a convolutional neural network. The major summary of the process from data preprocessing to network selection is shown in the `main.ipynb` file. I used the [Optuna library](https://optuna.org/) to search for optimal hyperparameters for the model. We intentionally refrained from using larger, prebuilt or pretrained models in order to get a feel for Pytorch and have a more hands-on learning experience. Because of this, our accuracy suffered from what may have been achieved with a more capable model. Future work on this project may include 
 - Trying out other, more capable model architectures
 - Tuning the spectrogram parameters
 - Experiment with adding in human-selected features from the GTZAN dataset along with the spectrogram analysis 

If you are interested in using the same dataset, please see the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

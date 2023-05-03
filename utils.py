import os
import numpy as np
from math import ceil
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import mirdata
import pydub
# from musicnn.extractor import extractor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model


def load_emoMusic(data_home):
    """
    Load track id and corresponding valence and arousal values of emoMusic dataset.

    Parameters
    ----------
    data_home: str
        Directory of data.

    Returns
    -------
    train_ids : list
        List of training sample ids.
    train_valence: np.ndarray
        Training valence values.
    train_arousal: np.ndarray
        Training arousal values.
    test_ids : list
        List of test sample ids.
    test_valence: np.ndarray
        Test valence values.
    test_arousal: np.ndarray
        Test arousal values.
    """
    annotations_path = data_home + 'emoMusic/annotations/static_annotations.csv'
    metadata_path = data_home + 'emoMusic/annotations/songs_info.csv'

    # Load track id for training and test
    with open(metadata_path) as f:
        reader = csv.reader(f, delimiter=',')
        train_ids = list()
        test_ids = list()
        for row in reader:
            if row[7] == 'development':
                train_ids.append(row[0])
            elif row[7] == 'evaluation':
                test_ids.append(row[0])

    # Load corresponding valence and arousal values
    with open(annotations_path) as f:
        reader = csv.reader(f, delimiter=',')
        train_arousal = list()
        train_valence = list()
        test_arousal = list()
        test_valence = list()
        for row in reader:
            if row[0] in train_ids:
                train_arousal.append(float(row[1]))
                train_valence.append(float(row[3]))
            elif row[0] in test_ids:
                test_arousal.append(float(row[1]))
                test_valence.append(float(row[3]))
    train_valence = np.array(train_valence, dtype=np.float32)
    train_arousal = np.array(train_arousal, dtype=np.float32)
    test_valence = np.array(test_valence, dtype=np.float32)
    test_arousal = np.array(test_arousal, dtype=np.float32)

    return train_ids, train_valence, train_arousal, test_ids, test_valence, test_arousal


def load_rwc_popular(data_home, save_clip=False):
    """
    Load rwc_popular dataset and split it into 10-second clips.

    Parameters
    ----------
    data_home: str
        Directory of data.
    save_clip: bool, optional
        Whether to save clip files, by default False.

    Returns
    -------
    clip_paths: list
        List of data paths.
    """
    rwc_popular = mirdata.initialize('rwc_popular', data_home=data_home + 'rwc_popular') # 49 tracks

    # Filter out tracks without audio file
    new_track_ids = list()
    for id in rwc_popular.track_ids:
        # if id in ['RM-P046', 'RM-P052', 'RM-P058', 'RM-P092']:
        #     continue
        try:
            audio = rwc_popular.track(id).audio
            new_track_ids.append(id)
        except:
            continue
    print("Number of tracks: ", len(new_track_ids))

    # Split each track into 10-second clips
    ten_seconds = 10 * 1000
    clip_paths = list()
    for id in new_track_ids:
        audio_path = rwc_popular.track(id).audio_path
        song = pydub.AudioSegment.from_wav(audio_path)
        duration = song.duration_seconds
        for i in range(ceil(duration/10)):
            clips = song[i * ten_seconds:(i + 1) * ten_seconds]
            if clips.duration_seconds < 3:
                continue
            if save_clip:
                clips.export(data_home + 'rwc_popular/clips_10seconds/' + id + '_' + str(i) + '.wav', format="wav")
            clip_paths.append(id + '_' + str(i) + '.wav')
    return clip_paths


def extract_features_emoMusic(paths, data_home, save_as=False):
    """
    Wrapper function for extracting features of dataset emoMusic.

    Parameters
    ----------
    paths : list
        List of audio paths
    data_home: str
        Directory of data.
    save_as : bool, optional
        Whether to save the features in the specified file, by default False.

    Returns
    -------
    X : np.ndarray
        Extracted features of audios in given paths.
    """
    # Extract features
    first_audio = True
    for p in paths:
        sound = pydub.AudioSegment.from_mp3(data_home + 'emoMusic/clips_45seconds/' + p + '.mp3')
        sound.export(data_home + 'emoMusic/clips_45seconds/' + p + '.wav', format="wav")
        taggram, tags, extracted_features = extractor(data_home + 'emoMusic/clips_45seconds/' + p + '.wav',
                                                      model='MSD_musicnn', extract_features=True, input_overlap=1)
        features = extracted_features['max_pool']
        if first_audio:
            X = features
            first_audio = False
        else:
            X = np.concatenate((X, features), axis=0)

    # Save data to file
    if save_as:
        # Create a directory where to store the extracted training features
        audio_representations_folder = data_home + 'emoMusic/audio_representations/'
        if not os.path.exists(audio_representations_folder):
            os.makedirs(audio_representations_folder)
        np.savez(audio_representations_folder + save_as, X=X)
        print('Audio features stored: ', save_as)

    return X


def extract_features_rwc(paths, data_home, save_as=False):
    """
    Wrapper function for extracting features of dataset rwc_popular.

    Parameters
    ----------
    paths : list
        List of audio paths
    data_home: str
        Directory of data.
    save_as : bool, optional
        Whether to save the features in the specified file, by default False.

    Returns
    -------
    X : np.ndarray
        Extracted features of audios in given paths.
    """
    # Extract features
    first_audio = True
    for p in tqdm(paths):
        taggram, tags, extracted_features = extractor(data_home + 'rwc_popular/clips_10seconds/' + p,
                                                      model='MSD_musicnn', extract_features=True, input_overlap=1)
        features = np.average(extracted_features['max_pool'], axis=0).reshape(1, -1)
        if first_audio:
            X = features
            first_audio = False
        else:
            X = np.concatenate((X, features), axis=0)

    # Save data to file
    if save_as:
        # Create a directory where to store the extracted training features
        audio_representations_folder = data_home + 'rwc_popular/audio_representations/'
        if not os.path.exists(audio_representations_folder):
            os.makedirs(audio_representations_folder)
        np.savez(audio_representations_folder + save_as, X=X)
        print('Audio features stored: ', save_as)

    return X


def data_generator(features, labels, shuffle=True):
    """
    Generator function that yields features and labels from the specified dataset.

    Parameters
    ----------
    features : np.ndarray
        Musicnn features of audios.
    labels : np.ndarray
        Ground truth of valence/arousal values.
    shuffle : bool, optional
        Whether to shuffle the data before iterating, by default True.

    Yields
    ------
    audio : np.ndarray
        A NumPy array containing the audio features.
    label : np.ndarray
        The corresponding label for the audio.
    """
    # Shuffle data
    if shuffle:
        idxs = np.random.permutation(len(labels))
        features = features[idxs]
        labels = labels[idxs]

    # Load features and label
    for idx in range(len(labels)):
        feature = np.average(features[idx], axis=0).reshape(1, -1)
        label = labels[idx]
        yield feature, label


def create_dataset(data_generator, input_args, input_shape, batch_size):
    """
    Create a TensorFlow dataset from a data generator function along with the specified input arguments and shape.

    Parameters
    ----------
    data_generator : callable
        The data generator function to use for creating the dataset.
    input_args : list
        A list containing the arguments to be passed to the data generator function.
    input_shape : tuple
        A tuple representing the shape of the input data.
    batch_size : int
        Batch size for the returned dataset.

    Returns
    -------
    dataset : tf.data.Dataset
        A TensorFlow dataset created from the data generator function.
    """
    dataset = tf.data.Dataset.from_generator(data_generator, args=input_args,
                                             output_signature=(
                                                 tf.TensorSpec(shape=input_shape, dtype=tf.float32),
                                                 tf.TensorSpec(shape=(), dtype=tf.float32)))
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


def data_generator_rwc(features):
    """
    Generator function that yields features from rwc_popular dataset.

    Parameters
    ----------
    features : np.ndarray
        Musicnn features of audios.

    Yields
    ------
    feature : np.ndarray
        A NumPy array containing the audio feature.
    """
    # Load features
    for idx in range(features.shape[0]):
        feature = features[idx].reshape(1, -1)
        yield feature


def create_dataset_rwc(data_generator, input_args, input_shape, batch_size):
    """
    Create a TensorFlow dataset from a data generator function along with the specified input arguments and shape.

    Parameters
    ----------
    data_generator : callable
        The data generator function to use for creating the dataset.
    input_args : list
        A list containing the arguments to be passed to the data generator function.
    input_shape : tuple
        A tuple representing the shape of the input data.
    batch_size : int
        Batch size for the returned dataset.

    Returns
    -------
    dataset : tf.data.Dataset
        A TensorFlow dataset created from the data generator function.
    """
    dataset = tf.data.Dataset.from_generator(data_generator, args=input_args,
                                             output_signature=(
                                                 tf.TensorSpec(shape=input_shape, dtype=tf.float32)))
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


def dnn_model(input_shape):
    """
    Build and compile a dense neural network model for emotion label prediction.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input tensor, excluding the batch size.

    Returns
    -------
    tf.keras.Model
        The compiled dense neural network model.

    Notes
    -----
    The dense neural network model consists of a flatten layer, a dense layer with 512 units, a dropout layer with 0.15
    dropout rate, and a output layer. The mean squared error loss function is used for training, and the Adam
    optimizer is used for optimization. The model is evaluated based on the RootMeanSquaredError metric during training.
    """
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.15)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss="mse",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def plot_loss(history):
    """
    Plot the training and validation loss and accuracy.

    Parameters
    ----------
    history : keras.callbacks.History
        The history object returned by the `fit` method of a Keras model.

    Returns
    -------
    None
    """

    rmse = history.history["root_mean_squared_error"]
    val_rmse = history.history["val_root_mean_squared_error"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(rmse) + 1)
    plt.plot(epochs, rmse, "bo", label="Training RMSE")
    plt.plot(epochs, val_rmse, "b", label="Validation RMSE")
    plt.title("Training and validation RMSE")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


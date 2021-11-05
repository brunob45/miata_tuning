import pandas as pd
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt


def build_and_compile_model(norm, rate=0.001):
    model = tf.keras.Sequential([
        norm,
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(rate))
    return model


def plot_loss(ax, history):
    if 'loss' in history.history.keys():
        ax.plot(history.history['loss'], label='loss')

    if 'val_loss' in history.history.keys():
        ax.plot(history.history['val_loss'], label='val_loss')

    ax.ylim([0, 1])
    ax.xlabel('Epoch')
    ax.ylabel('Error')
    ax.legend()
    ax.grid(True)


def plot_predict(ax, labels, predictions):
    a = ax.axes(aspect='equal')
    ax.scatter(labels, predictions)
    ax.xlabel('True Values')
    ax.ylabel('Predictions')
    lims = [0, 40]
    ax.xlim(lims)
    ax.ylim(lims)
    _ = ax.plot(lims, lims)


def split_dataset(dataset, label):
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    print(train_dataset.describe().transpose())

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop(label)
    test_labels = test_features.pop(label)

    return ((train_features, train_labels), (test_features, test_labels))


if __name__ == '__main__':
    filename = sys.argv[1] if len(sys.argv) > 1 else 'dataset.csv'
    dataset = pd.read_csv(filename, skiprows=2, sep='\t', decimal=',')

    dataset = dataset[['RPM', 'RPMdot', 'MAP', 'MAPdot', 'AFR',
                       'MAT', 'CLT', 'PW']]  # , 'SPK: Spark Advance' ]]

    # Split datasets in 2: for training and testing
    ((train_features, train_labels),
     (test_features, test_labels)) = split_dataset(dataset, 'PW')

    # Normalize datasets
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    print('Build and Compile')
    model = build_and_compile_model(normalizer)

    print('Train')
    history = model.fit(
        train_features,
        train_labels,
        # validation_split=0.2,
        # validation_data=(test_features, test_labels),
        verbose=2, epochs=120)

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # plot_loss(ax1, history)

    print('Evaluate')
    test_results = model.evaluate(test_features, test_labels, verbose=0)
    print(test_results)
    model.save('linear_model')

    print('Predict')
    test_predictions = model.predict(test_features).flatten()

    plot_predict(plt, test_labels, test_predictions)
    plt.show()
    print('Done')

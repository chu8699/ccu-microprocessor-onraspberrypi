# -*- coding: utf-8 -*-
"""
Created at 2019/12/8
@author: henk guo
tensorboard --logdir=logs/fit --port 8088 --host localhost

https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py

"""

import tensorflow as tf
import sklearn
import cv2
from sklearn.metrics import confusion_matrix
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io

from tensorflow import keras
import os
from model import resnet34
# os.environ["PATH"] += os.pathsep + 'D:\\app\\Graphviz\\bin\\'

def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    lr會在指定的epoch改變
    """
    learning_rate = 0.0005
    if epoch > 20:
        learning_rate = 0.0001
    with file_writer_lr.as_default():
        tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_confusion_matrix(cm, class_names, epoch):
    """
      Returns a matplotlib figure containing the plotted confusion matrix.
      Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
      """
    fig, ax = plt.subplots()

    im, cbar = heatmap(cm, class_names, class_names, ax=ax,
                       cmap="YlGn")
    texts = annotate_heatmap(im, valfmt="{x:.0f}")
    ax.set_title(f'epoch : {str(epoch).zfill(3)}')
    fig.tight_layout()
    fig.savefig(f'.\\cm\\output_{str(epoch).zfill(3)}.png')
    return fig


def log_confusion_matrix(epoch, logs):
    cm_sum = np.zeros((10, 10))
    for i in range(validation_steps//2):
        image_batch, label_batch = next(test_generator)
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = model.predict(image_batch)
        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        cm = confusion_matrix(label_batch, test_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        cm_sum += cm
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm_sum, class_names=class_names, epoch=epoch)
    cm_image = plot_to_image(figure)
    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


if __name__ == '__main__':
    logs_dir = "logs\\fit\\" + datetime.datetime.now().strftime("{}-%Y%m%d-%H%M%S".format('mfcc_cnn_tw'))
    class_names = [str(i) for i in range(10)]
    train_data_dir = './tw_data/img/'
    test_data_dir = './tw_data/test_img/'
    nrow = 250
    ncol = 250
    input_shape = (nrow, ncol, 3)
    batch_size_tr = 8
    batch_size_ts = 4
    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                           shear_range=0,
                                                           zoom_range=0,
                                                           horizontal_flip=False,
                                                           validation_split=0.1)

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(nrow, ncol),
        batch_size=batch_size_tr,
        class_mode='sparse',
        subset='training')

    test_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(nrow, ncol),
        batch_size=batch_size_ts,
        class_mode='sparse',
        subset='validation')

    model = keras.models.Sequential([
        keras.layers.Conv2D(16, (7, 7), activation='relu', padding='same', input_shape=input_shape),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),

        keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=input_shape),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),

        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),

        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),

        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),

        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),

        keras.layers.Flatten(),

        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(10, activation='softmax')
    ])

    # model = get_model()
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    steps_per_epoch = train_generator.n // batch_size_tr
    validation_steps = test_generator.n // batch_size_ts

    file_writer_cm = tf.summary.create_file_writer(logs_dir + '/cm')
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    file_writer_lr = tf.summary.create_file_writer(logs_dir + "/lr")
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    logs_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)

    model_filepath = "mfcc_cnn_model_all_tw.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                                    mode='max')


    model.fit_generator(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=30,
                        verbose=1,
                        validation_data=test_generator,
                        validation_steps=validation_steps,
                        callbacks=[logs_callback, checkpoint, cm_callback, lr_callback])


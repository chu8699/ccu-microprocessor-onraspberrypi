# -*- coding: utf-8 -*-
"""
Created at 2019/12/17
@author: henk guo
"""
import tensorflow as tf
model = tf.keras.models.load_model('mfcc_cnn_model_all_tw.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("mfcc_cnn_model_all_tw.tflite", "wb").write(tflite_model)

import numpy as np # linear algebra
import tensorflow as tf # TensorFlow
import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory # data processing
from tensorflow.keras.layers import RandomFlip, RandomTranslation, RandomRotation, RandomZoom, RandomContrast, RandomCrop, Dense, Average, Flatten, Dropout # layers used in NN
from keras.applications import xception, inception_resnet_v2, resnet_v2 # models used for transfer learning
import gc #garbage collection


# General variables
# seed for reproducible results (y, the first letter of my first name, in Ascii ;P )
my_seed = 121
# the directory in which the data is
dir = "../input/glasses-data/glasses"
# the directory where the data output goes
out_dir = "./"
# directory of images with glasses
glass_dir = dir + "/glasses/"
no_glass_dir = dir + "/no_glasses/"
# a bit bigger than final input to the model to augment the data a bit better
img_data_size = (512, 512)
img_size = (224, 224,3)
# batch size
batch_size = 16
# class weights, as the dataset is slightly imbalanced
weights = {0:0.44,1:0.56}


# method that returns custom callbacks, according to your model name and "base" or "tune" (base training or finetuning)
def get_callbacks(mode, name):
    if mode == "base":
        return [keras.callbacks.ReduceLROnPlateau(patience=2), keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True), keras.callbacks.TensorBoard(log_dir=f'{out_dir}{name}-logs-base'), tf.keras.callbacks.ModelCheckpoint(filepath=f"/{name}-best-base", monitor='val_loss', save_best_only=True)]
    else:
        return [keras.callbacks.ReduceLROnPlateau(patience=1), keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True), keras.callbacks.TensorBoard(log_dir=f'{out_dir}{name}-logs-tune'), tf.keras.callbacks.ModelCheckpoint(filepath=f"/{name}-best-tune", monitor='val_loss', save_best_only=True)]


# use keras preprocessing fucntionalities and get dataset
train = image_dataset_from_directory(directory=dir,  labels="inferred", label_mode="binary",
                                     image_size = img_data_size,  seed=my_seed,
                                     validation_split=0.2, subset="training", batch_size=batch_size)
validation = image_dataset_from_directory(directory=dir,  labels="inferred", label_mode="binary",
                                          image_size = img_data_size,  seed=my_seed,
                                          validation_split=0.2, subset="validation", batch_size=batch_size)


# function that returns a model of one of the models used for transfer learning (given the base model), removes code redundancy
def get_model(input_layer, preprocess, model):
    data_pre = preprocess(input_layer)
    name = model.name
    model.trainable = False
    base_model = model(data_pre)
    flat = Flatten()(base_model)
    drop1 = Dropout(rate=0.5, seed=my_seed)(flat)
    dense1 = Dense(units=128, activation="relu")(drop1)
    drop2 = Dropout(rate=0.2, seed=my_seed)(dense1)
    out = Dense(units=1, activation="sigmoid")(drop2)
    ret_model = keras.Model(input_tensor, out)
    ret_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
    print("start: base training " + name)
    ret_model.fit(train, epochs=20, verbose=2, validation_data=validation, shuffle=True, class_weight=weights, callbacks=get_callbacks("base", model.name))
    print("finished: base training " + name)
    # fine tuning
    ret_model.trainable = True
    ret_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=tf.keras.losses.BinaryCrossentropy(),metrics=["accuracy"])
    print("start: fine tuning " + name)
    ret_model.fit(train, epochs = 5, verbose=2, validation_data=validation,  shuffle=True, class_weight=weights, callbacks=get_callbacks("tune", model.name))
    print("finished: fine tuning " + name)
    # remove top two layers
    ret_model = keras.Model(input_tensor, dense1)
    # remove the last two layers from memory
    del drop2
    del out
    gc.collect()
    # make the model not trainable
    ret_model.trainable = False
    return ret_model, dense1


# initialize input layer(s)
input_tensor = keras.Input(shape=(512,512,3))
data_flip = RandomFlip(mode="horizontal", seed=my_seed)(input_tensor)
data_trans = RandomTranslation(height_factor=0.1, width_factor=0.1, seed=my_seed)(data_flip)
data_rot = RandomRotation(factor=0.25, seed=my_seed)(data_trans)
data_zoom = RandomZoom(height_factor=0.1, width_factor=0.1)(data_rot)
data_contrast = RandomContrast(factor=0.3, seed=my_seed)(data_zoom)
data_pre = RandomCrop(height=224, width=224, seed=my_seed)(data_contrast)


# get the three base models
xception_model, xception_layer = get_model(data_pre, xception.preprocess_input, xception.Xception(include_top=False, weights = "imagenet", input_shape=img_size))
inception_resnet, inception_layer = get_model(data_pre, inception_resnet_v2.preprocess_input, inception_resnet_v2.InceptionResNetV2(include_top=False, weights = "imagenet", input_shape=img_size))
resnet_model, resnet_layer = get_model(data_pre, resnet_v2.preprocess_input, resnet_v2.ResNet152V2(input_shape = img_size, include_top=False, weights = "imagenet"))
# garbage collection
gc.collect()


# create ensemble model
concat = tf.keras.layers.Concatenate()([xception_layer, inception_layer, resnet_layer])
drop1 = Dropout(rate=0.5, seed=my_seed)(concat)
dense = Dense(units=128, activation="relu")(drop1)
drop2 = Dropout(rate=0.2, seed=my_seed)(dense)
out = Dense(units=1, activation="sigmoid")(drop2)
ensemble = keras.Model(input_tensor, out)


# compile the model
ensemble.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
ensemble.summary()
# fit the model
print("start ensemble training")
ensemble.fit(train, epochs=20, verbose=2, validation_data=validation, shuffle=True, class_weight=weights, callbacks=get_callbacks("base", "ensemble"))
print("done ensemble training")
ensemble.save(f"{out_dir}final")
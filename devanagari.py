import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
#from keras.callbacks import TensorBoard
#from keras.preprocessing.image import ImageDataGenerator


# Import the data
dataset = pd.read_csv("data.csv")

# Check the "head" of the dataframe
dataset.head()
x = dataset.values[:,:-1] / 255.0
y = dataset['character'].values
# Free memory
del dataset

n_classes = 46

# Let's split the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Encode the categories
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

# Now let's visualise one image
img_width_cols = 32
img_height_rows = 32
idx = 50
image = x_train[idx, :].reshape((img_width_cols, img_height_rows)) * 255
plt.imshow(image.astype(np.uint8))
# plt.title(y_train[idx])
# plt.show()

# Now we shall build the CNN!
# We need to reshape all the data however
im_shape = (img_height_rows, img_width_cols, 1)
x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)

# Let's define a model. Here we shall use a Sequential model.
cnn = Sequential()

# Now let's define the layers. the first layer would be the input convolutional layer
# Because its the input layer we also have to specify the input size its looking for
n_filters = 32
kernelSize = (3, 3)
ip_activation = 'relu'
ip_conv_0 = Conv2D(filters=n_filters, kernel_size=kernelSize, input_shape=im_shape, activation=ip_activation)
cnn.add(ip_conv_0)
ip_conv_0_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_0_1)

pool_0 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_0)

ip_conv_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_1)
ip_conv_1_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_1_1)

pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_1)

drop_layer_0 = Dropout(0.2)
cnn.add(drop_layer_0)

flat_layer_0 = Flatten()
cnn.add(Flatten())

# Now add the Dense layers
h_dense_0 = Dense(units=128, activation=ip_activation, kernel_initializer='uniform')
cnn.add(h_dense_0)

# drop_layer_1 = Dropout(0.1)
# cnn.add(drop_layer_1)

h_dense_0 = Dense(units=64, activation=ip_activation, kernel_initializer='uniform')
cnn.add(h_dense_0)


op_activation = 'softmax'
output_layer = Dense(units=n_classes, activation=op_activation, kernel_initializer='uniform')
cnn.add(output_layer)

# Now we define the optimizer and loss functions to minimize
opt = 'adam'#Adam(lr=0.001)
loss = 'categorical_crossentropy'
metrics = ['accuracy']

# Compile the classifier using the configuration we want
cnn.compile(optimizer=opt, loss=loss, metrics=metrics)

# Now we finally fit and train the model
history = cnn.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test))

# Evaluate this model
scores = cnn.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# summarize history for accuracy
fig1, ax_acc = plt.subplots()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()
# summarize history for loss
fig2, ax_loss = plt.subplots()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# serialize model to JSON
model_json = cnn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn.save_weights("model_weights.h5")
print("Saved model to disk")


# load json and create model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_weights.h5")
print("Loaded model from disk")
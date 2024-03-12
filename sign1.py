
               #custom model
import scipy
import numpy as np
import pandas as pd  # data procesing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
import matplotlib.pyplot as plt

# training = pd.read_csv("C:\\Users\\udayk\Downloads\\archive (8)\sign_mnist_train.csv")
# testing = pd.read_csv("C:\\Users\\udayk\Downloads\\archive (8)\sign_mnist_test.csv")
# # training_images = np.expand_dims(training_images, axis=3)
# # testing_images = np.expand_dims(testing_images, axis=3)
# training_images = training.iloc[:, 1:].values
# training_labels = training.iloc[:, 0].values
# testing_images = testing.iloc[:, 1:].values
# testing_labels = testing.iloc[:, 0].values
# training_images = training_images.reshape(-1, 28, 28, 1)
# testing_images = testing_images.reshape(-1, 28, 28, 1)
# # print(training_images.shape)
# # print(training_labels.shape)
# # print(testing_images.shape)
# # print(testing_labels.shape)


# figure,axis=plt.subplots(3,4)
# # a,b=plt.subplots(5,10)
# figure.set_size_inches(10, 10)
# k = 0
# for i in range(3):
#     for j in range(4):
#         axis[i,j].imshow(training_images[k].reshape(28, 28) , cmap = "gray")   #imshow() is used to display the image
#         k += 1                                                                  #k is used to iterate through the images
#     # plt.tight_layout()                                                      #tight_layout() is used to adjust the subplots to fit into the figure area
# plt.show()                                                                    #show() is used to display the figure
# #it is the function to initialze the data augummentation

# train_img_gen=ImageDataGenerator(
#     # blurring=0,
#     rescale=1/255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode="nearest",
#     # adaptive_equalization=True
# )
# valid_data= ImageDataGenerator(
#     rescale=1 / 255
# )

# # Keep These
# print(training_images.shape)
# print(testing_images.shape)

# # training the network model using the data augmentation
# model = keras.models.Sequential([
#     keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     keras.layers.MaxPooling2D(2, 2),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(2, 2),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(2, 2),

#     keras.layers.Flatten(),
#     keras.layers.Dense(512, activation='relu'),
#     # keras.layers.Dense(512, activation='relu'),
#     keras.layers.Dense(26, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # train the model
# h=model.fit_generator(train_img_gen.flow(training_images,training_labels,batch_size=32),steps_per_epoch=len(training_images)/32,epochs=10,validation_data=valid_data.flow(testing_images,testing_labels,batch_size=32),validation_steps=len(testing_images)/32)
# model.evaluate(testing_images, testing_labels, verbose=0)
# print(h.history.keys())


                                            LENET 5 model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import scipy

# Read the data
training = pd.read_csv("C:\\Users\\udayk\Downloads\\archive (8)\sign_mnist_train.csv")
testing = pd.read_csv("C:\\Users\\udayk\Downloads\\archive (8)\sign_mnist_test.csv")

# Extract images and labels
training_images = training.iloc[:, 1:].values  # 20 percent for training
training_labels = training.iloc[:, 0].values
testing_images = testing.iloc[:, 1:].values   # 80 percent for validation
testing_labels = testing.iloc[:, 0].values

# Reshape images
training_images = training_images.reshape(-1, 28, 28, 1)
testing_images = testing_images.reshape(-1, 28, 28, 1)

# Display sample images
figure, axis = plt.subplots(3, 4)
figure.set_size_inches(10, 10)
k = 0
for i in range(3):
    for j in range(4):
        axis[i,j].imshow(training_images[k].reshape(28, 28) , cmap = "gray")
        k += 1
    plt.tight_layout()
plt.show()

# Data augmentation for training images
train_img_gen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Data augmentation for validation/testing images
valid_data = ImageDataGenerator(
    rescale=1 / 255
)

# Print shapes of data
print(training_images.shape)
print(testing_images.shape)

# Define LeNet model
model = keras.models.Sequential([
    keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(16, (5, 5), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation='relu'),
    keras.layers.Dense(84, activation='relu'),
    keras.layers.Dense(26, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]


# Train the model
h = model.fit_generator(train_img_gen.flow(training_images, training_labels, batch_size=32),
                        steps_per_epoch=len(training_images)/32,
                        epochs=10,
                        # validation_split=0.2,
                        # callbacks=callback,
                        validation_data=valid_data.flow(testing_images, testing_labels, batch_size=32),
                        validation_steps=len(testing_images)/32)

# Evaluate the model
model.evaluate(testing_images, testing_labels, verbose=0)
classes = ["Class " + str(i) for i in range(26) if i != 9]
# Predictions
# Predictions
# Suppress warnings
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Predictions
predictions = model.predict(testing_images)
adjusted_predictions = []
for i in range(len(predictions)):
    max_class_index = np.argmax(predictions[i])
    if max_class_index >= 9:
        max_class_index += 1
    adjusted_predictions.append(max_class_index)

# Print adjusted predictions
print(adjusted_predictions[:5])
import seaborn as sns
# Precision, recall, f1-score for all the classes
print(classification_report(testing_labels, adjusted_predictions, target_names=classes))

# Confusion matrix for the model predictions
cm = confusion_matrix(testing_labels, adjusted_predictions)
print("Confusion Matrix:")
print(cm)



# lennet5 model accuracy is loss: 0.2793 - accuracy: 0.9072 - val_loss: 0.2067 - val_accuracy: 0.9307
#custom model accuracy is loss: 0.5872 - accuracy: 0.8045 - val_loss: 0.2584 - val_accuracy: 0.9090

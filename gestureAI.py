# ============================== #
#       📦 Import Libraries      #
# ============================== #
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os

# ============================== #
#       🔧 Set Parameters        #
# ============================== #
img_height, img_width = 64, 64
batch_size = 32
epochs = 10

train_dir = 'dataset\\train'
val_dir = 'dataset\\test'

# ============================== #
#   🧪 Check Class Imbalance     #
# ============================== #
print("\n📊 Class image distribution:")
for label in os.listdir(train_dir):
    path = os.path.join(train_dir, label)
    if os.path.isdir(path):
        print(f" - {label}: {len(os.listdir(path))} images")

# ============================== #
#   📂 Data Loading & Augment    #
# ============================== #
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=15
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = train_generator.num_classes

# ============================== #
#   💾 Save Class Labels         #
# ============================== #
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)
print("\n✅ Saved class mapping:", train_generator.class_indices)

# ============================== #
#       🧠 Build Model           #
# ============================== #
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# ============================== #
#       ⚙️ Compile Model         #
# ============================== #
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ============================== #
#       🏋️ Train Model           #
# ============================== #
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# ============================== #
#   📈 Accuracy and Loss Graphs  #
# ============================== #
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('📊 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('📉 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# ============================== #
#      💾 Save Trained Model     #
# ============================== #
model.save("gesture_model.h5")
print("\n✅ Model saved as gesture_model.h5")

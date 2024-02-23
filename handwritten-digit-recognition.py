import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data() 

X_train = tf.keras.utils.normalize(X_train)
X_test = tf.keras.utils.normalize(X_test)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
model.fit(X_train, y_train, epochs=3)
 
model.save('handwritten.model')


model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(X_test, y_test)
 
print(loss)
print(accuracy)


image_number = 1

while os.path.isfile(f'digits/digit{image_number}.png'):
    try:
        img = cv2.imread(f'digits/digit{image_number}.png')[:,:,0]
        img = np.invert(img)
        prediction = model.predict(img)
        print(f'This digit is probably a {np.argmax(prediction)}')
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print()
    finally:
        image_number += 1
    
    
    
    
    
    
    
    
    
    
    
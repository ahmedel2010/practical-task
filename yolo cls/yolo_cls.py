from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
model = YOLO("yolov8n-cls.pt")

model.train(data='Dataset',epochs=20)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
dataGenerator = ImageDataGenerator(
    rescale = 1/255
)

TestGenerator= dataGenerator.flow_from_directory(directory= 'Dataset/test',target_size=(176,208),batch_size= 16, class_mode='sparse',shuffle=True)


random = next(TestGenerator)

random_image = random[0][0]

random_image = (random_image * 255).astype(np.uint8)

image_PIL = Image.fromarray(random_image)

results = model.predict(source= image_PIL)

predicted = results[0].probs.data


predicted_class = predicted.argmax()

class_name =results[0].names[predicted_class.item()]


plt.imshow(image_PIL.resize((144,144))) 
plt.title(class_name)
plt.show()

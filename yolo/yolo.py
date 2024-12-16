
from ultralytics import YOLO
import matplotlib.pyplot as plt
model = YOLO("yolo11m.pt")
path = "9.jpg"
results = model(source=path,conf=0.4)
result_img = results[0].plot()  
plt.imshow(result_img)
plt.axis('off')
plt.show()
    

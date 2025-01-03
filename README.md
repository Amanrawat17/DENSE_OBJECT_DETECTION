# Dense Object Detection
Object Detection in Dense Environments using Yolov5 by Ultralytics on SKU110K dataset and Post Quantization.



### The Yolov5 model by Ultralytics is an efficient object detection model that can be quantized and deployed on edge devices for real-time detection in dense environments.
Some of the key features of Yolov5 are:
- It is fast (30+ FPS) and accurate.
- It has small model size (from 3.2MB to 253MB) which makes it suitable for deployment on edge devices.
- It provides several versions (yolov5s, yolov5m, yolov5l, yolov5x) for different accuracy-speed tradeoffs.
- It has an active development community which provides frequent updates.
- It has PyTorch, TensorFlow and ONNX versions which provides flexibility in the choice of framework.
- It can be quantized to INT8 precision with minimal loss in accuracy making it suitable for edge devices with low memory and compute.

### SKU-110K Dataset for Object Detection
The SKU-110K dataset is a large-scale dataset for object detection tasks. The dataset was released by SenseTime, a leading AI company, and is designed to enable researchers and developers to train and test object detection models.
Some of the key features of the dataset are:
- It has 110,000 images with 1,000,000 objects.
- It contains a large number of images with high-quality annotations, which makes it ideal for training deep learning models.
- It has a large number of objects per image, which makes it suitable for training models for dense object detection scenarios.

## Training and Evaluation
The training was done on SKU110K dataset for a maximum of 50 epochs but during the training we could see the metrics saturate around 30 epochs when the mAP50 was almost 0.6 on the training set.
Yolov5x which is the largest version of the model was used for training and evaluation from scratch. (No pretrained weights were used)
- The training set consisted of 8185 images.
- The validation set consisted of 584 images.
- The test set consisted of 2920 images.

The image size was set to 640x640 and the batch size was set to 2 due to the data being large and avoiding GPUs running out of memory.
The training was done on 2 x 3060 Ti Nvidia GPUs each consisting of 8gb vram for approximately 3 hours.



## Results
- I was able to achieve exceptional results with the Yolov5 model on the SKU110K dataset
- But remember the main objective of this project was to quantize the model and deploy it on edge devices for real-time detection in dense environments.
- The model was quantized into various formats and different precisions and the results were compared to one another on test set.

| Index | Model                         | Size (mb) | Precision | Recall | mAP50 | F1 Score |
| ------| ----------------------------- | --------- | --------- | ------ | ----- | -------- |
| 1     | Pytorch (.pt)                 | 164       | 0.930     | 0.868  | 0.922 | 0.8979   |
| 2     | Tensorflow (.pb)              | 329       | 0.925     | 0.868  | 0.919 | 0.8955   |
| 3     | Tflite Float 32 (.tflite)     | 328       | 0.925     | 0.868  | 0.919 | 0.8955   |
| 4     | Tflite Float 16 (.tflite)     | 164       | 0.925     | 0.868  | 0.919 | 0.8955   |
| 5     | Tflite Int 8 (.tflite)        | 83        | 0.917     | 0.865  | 0.915 | 0.8902   |


> **_NOTE:_** Individual results and plots for each model can be found in the repository in test folder.

## Inference
- As we can see the models in the middle perform very similar to one another.
- Here are some inference results for various models.

Theoritically the model should have been quantized to INT8 precision with minimal loss in accuracy and we could see that the model was quantized to INT8 precision with a loss in accuracy of 0.007 which is very minimal and works in our favor.

But, when these models were inferenced and compared to one another the int 8 model performs the best in terms of accuracy.
Amazing right? :)

## Future (Real World Implementations)
- Inventory Management: Retail stores can use object detection models to automatically track and manage inventory levels for different products. By analyzing product images captured in real-time, the YOLOv5 model can detect and count the number of products on shelves or in storage, and alert store managers when inventory levels are running low.
- Loss Prevention: Object detection models can also be used to identify and prevent theft or fraud in retail stores. By analyzing surveillance camera footage or product images, the YOLOv5 model can detect and track suspicious behavior, such as shoplifting or product tampering, and alert store security personnel.
- Customer Behavior Analysis: Retail stores can use object detection models to analyze customer behavior and preferences. By analyzing product images and tracking customer movements, the YOLOv5 model can identify popular product categories and display placements, and provide insights into customer shopping behavior.
- Product Recommendation: Object detection models can also be used to provide personalized product recommendations to customers. By analyzing customer images or product images captured in real-time, the YOLOv5 model can identify the products that a customer is interested in and recommend complementary or similar products.



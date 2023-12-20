from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import tensorflow as tf



# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("/Users/fyoung24/Downloads/converted_keras (1)/keras_model.h5", compile=False)

# Load the labels
class_names = open("/Users/fyoung24/Downloads/converted_keras (1)/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Open a connection to the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the image to be at least 224x224 and then crop from the center
    image = Image.fromarray(image)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the prediction and confidence score on the frame
    cv2.putText(frame, f"Class: {class_name[2:]}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence Score: {confidence_score:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Live Object Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

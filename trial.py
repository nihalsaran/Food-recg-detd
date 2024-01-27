import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="test.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Load class labels
class_labels = ["almond", "apple","banana","cashew", "club_sandwich", "cup_cakes", "donuts", "fig", "french_fries", "ice_cream", "nachos", "omelette", "orange", "pizza", "raisin", "samosa", "spring_rolls", "tacos"]
'', '', '', 

while True:
    ret, frame = cap.read()

    # Pre-process the frame
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Perform inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process the output tensor
    for detection in output_data[0]:
        score = detection[2]
        class_id = int(detection[1])

        if score > 0.5:
            # Extract bounding box coordinates
            left = int(detection[3] * frame.shape[1])
            top = int(detection[4] * frame.shape[0])
            right = int(detection[5] * frame.shape[1])
            bottom = int(detection[6] * frame.shape[0])

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display class label and confidence score
            label = "{}: {:.2f}%".format(class_labels[class_id], score * 100)
            cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

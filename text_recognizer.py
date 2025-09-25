import cv2
import easyocr
import pyttsx3
import numpy as np

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Specify the language for OCR
reader = easyocr.Reader(['en'])

def read_text(image):
    # Use EasyOCR to perform OCR on the image
    result = reader.readtext(image)

    # Extract and concatenate the recognized text
    text = ' '.join([box[1] for box in result])

    return text, result

def speak_text(text):
    # Use the text-to-speech engine to read the text aloud
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

def draw_bounding_boxes(image, result):
    for box in result:
        # Extract the bounding box coordinates
        points = np.array(box[0], dtype=np.int32)  # Convert to NumPy array
        points = points.reshape((-1, 1, 2))

        # Draw the bounding box around the text
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Put the recognized text next to the bounding box
        cv2.putText(image, box[1], (points[0][0][0], points[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def text_reader():
    # Open the video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
        # Read a frame from the video capture
    ret, frame = cap.read()

        # Perform OCR on the frame
    text, result = read_text(frame)

        # Draw bounding boxes and display text on the frame
    draw_bounding_boxes(frame, result)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the modified frame
    cv2.imshow('Live Video Capture', frame)

        # Speak the recognized text
    speak_text(text)

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    read_text()

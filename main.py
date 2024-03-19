# Importing the necessary libraries
import numpy as np
import cv2
import pytesseract
from PIL import Image
import ssl
import os
import time
from tensorflow.lite.python.interpreter import Interpreter
from concurrent.futures import ThreadPoolExecutor
import re
import requests
from flask import Flask, Response, render_template, jsonify

# Providing the API KEY used with the service
API_KEY = '2c82b7c5-260a-42fb-828c-3961436b6272'
# API Endpoint URL that will be used to get vehicle information
API_URL = 'https://www.vegvesen.no/ws/no/vegvesen/kjoretoy/felles/datautlevering/enkeltoppslag/kjoretoydata?kjennemerke='

# Definition of Vehicle class to hold vehicle-specific data
class Vehicle:
    def __init__(self, data):
        # If data is None, no further processing is done in the constructor
        if data is None:
            return
        # Parsing relevant fields from input data
        self.license_plate = data['kjoretoydataListe'][0]['kjoretoyId'].get('kjennemerke')
        self.type = data['kjoretoydataListe'][0]['godkjenning']['tekniskGodkjenning']['tekniskeData']['generelt']['tekniskKode'].get('kodeNavn')
        self.brand = data['kjoretoydataListe'][0]['godkjenning']['tekniskGodkjenning']['tekniskeData']['generelt']['merke'][0]['merke']
        self.fuel = data['kjoretoydataListe'][0]['godkjenning']['tekniskGodkjenning']['tekniskeData']['miljodata']['miljoOgdrivstoffGruppe'][0]['drivstoffKodeMiljodata'].get('kodeNavn')
        self.drivetrain = data['kjoretoydataListe'][0]['godkjenning']['tekniskGodkjenning']['tekniskeData']['motorOgDrivverk']['girkassetype'].get('kodeBeskrivelse')
        self.max_speed = data['kjoretoydataListe'][0]['godkjenning']['tekniskGodkjenning']['tekniskeData']['motorOgDrivverk']['maksimumHastighet'][0]
        self.power = str(int(data['kjoretoydataListe'][0]['godkjenning']['tekniskGodkjenning']['tekniskeData']['motorOgDrivverk']['motor'][0]['drivstoff'][0]['maksNettoEffekt'])) + ' hp'
        self.weight = str(data['kjoretoydataListe'][0]['godkjenning']['tekniskGodkjenning']['tekniskeData']['vekter'].get('egenvekt')) + ' kg'
        self.first_registration = data['kjoretoydataListe'][0]['forstegangsregistrering'].get('registrertForstegangNorgeDato')

# Function that makes a GET request to the built URL and returns JSON response if successful
def get_vehicle_info(license_plate):
    headers = {
        'SVV-Authorization': f'Apikey {API_KEY}' # Inserting API key into header
    }
    response = requests.get(API_URL + license_plate, headers=headers)
    if response.status_code == 200: # HTTP OK status
        return response.json()
    else:
        return {}

ssl._create_default_https_context = ssl._create_unverified_context

# Define paths and constants for the object detection model
modelpath=f'detect.tflite' # Path to trained TensorFlow Lite model
lblpath='labelmap.txt' # Path to label map file
min_conf=0.3 # Minimum confidence value for detected objects
cap = cv2.VideoCapture(0) # Open default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 992) # Camera frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 558) # Camera frame height

# Load the TFLite model and allocate tensors
interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()

# Get input and output tensors and their shapes
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
float_input = (input_details[0]['dtype'] == np.float32)

# Normalization parameters for the input image
input_mean = 127.5
input_std = 127.5

# Read the label map file
with open(lblpath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Function to process the image and call the OCR method
def ocr_process(license_plate_img):
    # Deblur the image
    license_plate_img = cv2.GaussianBlur(license_plate_img, (0, 0), 3)
    license_plate_img = cv2.addWeighted(license_plate_img, 1.5, license_plate_img, -0.5, 0)

    # Remove the leftmost 15%, topmost 2%, bottommost 2% and rightmost 2% of the image
    height, width = license_plate_img.shape[:2]
    license_plate_img = license_plate_img[int(height*0.01):int(height*0.99), int(width*0.15):int(width*0.98)]

    # Convert the image to grayscale
    gray = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

    # Apply a blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert the image to make the text black and the background white
    inverted = cv2.bitwise_not(thresh)

    # Dilate the image to make the text more clear
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(inverted, kernel, iterations=1)

    # Use pytesseract to recognize the license plate number
    custom_config = r'--oem 3 --psm 6'
    result = pytesseract.image_to_string(Image.fromarray(dilated), config=custom_config)

    return result

# Function to format the recognized license plate text
def license_plate_formatter(string):
    # Find two letters followed by five digits
    match = re.search(r'([a-zA-Z]{2}).*?(\d{5})', string)

    if match:
        # Extract the letters and digits
        letters = match.group(1)
        digits = match.group(2)

        # Capitalize the letters and concatenate with the digits
        formatted_string = letters.upper() + digits
    else:
        formatted_string = None

    return formatted_string

# Setup the Flask application
app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

newest_vehicle = None

# Function to generate video feed frames, detect and recognize license plates, call the API with recognized value
def generate_frames():
    global newest_vehicle
    license_plate_counts = {}
    license_plate = "none"
    last_ocr_time = 0
    last_license_plate_text = None
    last_detection_time = 0
    detection_interval = 1  # seconds
    ocr_interval = 1  # seconds
    last_ocr_time = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        while(True):

            ret, frame =cap.read()

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imH, imW, _ = frame.shape
            image_resized = cv2.resize(image_rgb, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)
            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if float_input:
                input_data = (np.float32(input_data) - input_mean) / input_std
                
            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()
            
            boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
            
            for i in range(len(scores)):
                if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    object_name = labels[int(classes[i])] 
                    label = '%s: %d%%' % (license_plate, int(scores[i]*100)) 
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
                    label_ymin = max(ymin, labelSize[1] + 10) 
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 
                    
                    # Crop the license plate image
                    license_plate_img = frame[ymin:ymax, xmin:xmax]
                    current_time = time.time()
                    if current_time - last_ocr_time > ocr_interval:
                        license_plate_text = executor.submit(ocr_process, license_plate_img)
                        last_ocr_time = current_time

                    # Only print the license plate text if it's different from the last one and enough time has passed
                    if license_plate_text != last_license_plate_text and current_time - last_detection_time > detection_interval:
                        license_plate = license_plate_formatter(license_plate_text.result())
                        if license_plate:
                            # Increment the count for this license plate
                            if license_plate in license_plate_counts:
                                license_plate_counts[license_plate] += 1
                            else:
                                license_plate_counts[license_plate] = 1

                            # Only make the API call if this license plate has been detected three times
                            if license_plate_counts[license_plate] >= 3 and (not newest_vehicle or license_plate != newest_vehicle.license_plate):
                                vehicle_info = get_vehicle_info(license_plate)
                                if vehicle_info:  # Check if vehicle_info is not empty
                                    newest_vehicle = Vehicle(vehicle_info)
                                    print(newest_vehicle.__dict__)
                                last_license_plate_text = license_plate
                                last_detection_time = current_time

            if ret:
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                break

@app.route('/')
def index():
    global newest_vehicle
    if newest_vehicle:
        return render_template('index.html', vehicle=newest_vehicle)
    else:
        return render_template('index.html', vehicle=None)

@app.route('/video_feed')
def video_feed():
    # returning video feed over HTTP
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_vehicle')
def get_vehicle():
    # returning vehicle data as JSON
    global newest_vehicle
    if newest_vehicle:
        return jsonify(newest_vehicle.__dict__) 
    else:
        return jsonify({})
    
@app.route('/reset_vehicle')
def reset_vehicle():
    # endpoint to manually reset the newest_vehicle variable
    global newest_vehicle
    newest_vehicle = None
    return jsonify({})

# Main segment which starts Flask app
if __name__ == '__main__':
    app.run(debug=True, port=8080)

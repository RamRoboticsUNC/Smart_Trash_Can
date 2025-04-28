# Smart_Trash_Can

Maria:
Github Link: https://github.com/chungchunwang/Smartsort

Follow the instructions at the bottom of this link. 

Extra Steps I took:
To set up Firebase:
1. Go to Firebase Console
2. Click "Add project" and follow the setup process.
3. Once created, navigate to Build > Realtime Database and Create a Database.
4. Set the stuff in Rules to true like this:
{
  "rules": {
    ".read": true,
    ".write": true
  }
}

5. Go to Realtime Database, Click the "Data" tab, Copy the URL and paste it into main.py where it says: url = 'swap...'
6. In src/firebase.js, replace all the code in it with your config stuff. You can get it all the code at Project Overview -> make an app and then click on it

7. Run: pip install --user torch torchvision fastapi uvicorn requests numpy pillow opencv-python pickle-mixin
8. uvicorn main:app --reload
(for me, I had to uninstall, reinstall, and upgrade a bunch of libraries in order to get it working)

9. Once it works, go to http://127.0.0.1:8000/docs to test it out

10. In main.py: add these print and else statements in these areas for clarity in the console output:

model = inception_v3(pretrained=True)
print("model eval")
model.eval()
if torch.cuda.is_available():
    model.to('cuda')
else: 
    print("gpu not avilalable")


    

@app.post("/")
async def root(weight: str, file: UploadFile = File(...)):
    print("Actually start testing the model")

  with torch.no_grad():
        output = model(input_batch)
        print("you reached the goal")
        print(output)


        

if data[idx] in recyclable:
    recycle = 1
    print(f"The object {data[idx]} is recyclable.")
else:
    print(f"The object {data[idx]} is not recyclable.")

---

Ezra:

Github link: https://github.com/manuelamc14/waste-classification-model

Follow instructions in this repo's README to install, including changing the
file structure and file paths in `main.py` and `index.py`.

Make sure to set up a virtual environment and run `pip install -r requirements.txt`.

We were running `main.py` instead of `index.py` like the instructions said, so
use `python3 main.py`. You will probably be prompted to download packages even after you
install the requirements.txt, just follow what the errors say.

Once we started getting errors that no longer prompted us to install packages, we started messing with the main.py file. 

You can copy and paste this main.py file, but make sure to keep your correct paths on your machine (lines 14 & 15).

```python
# Dependencies
import numpy as np
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
import tensorflow as tf

# def predict(img_path):
def getPrediction(filename):
     model = tf.keras.models.load_model("/Users/ezraheinberg/Projects/smart_trash_can_project/waste-classification-model/Resources/Model/final_model_weights.hdf5")
     img = load_img('/Users/ezraheinberg/Projects/smart_trash_can_project/waste-classification-model/static/'+filename, target_size=(180, 180))
     img = img_to_array(img)
     img = img / 255
     img = np.expand_dims(img,axis=0)
     #category = model.predict_classes(img)
     category = model.predict_step(img)
     answer = category[0]
     print(answer)
     probability = model.predict(img)
     probability_results = 0

     is_first_greater = tf.math.greater(answer[0], answer[1])
     print(is_first_greater)
     if is_first_greater:
          answer = "Recyclable"
          probability_results = probability[0][1]
     else:
          answer = "Organic"
          probability_results = probability[0][0]

     answer = str(answer)
     probability_results=str(probability_results)

     values = [answer, probability_results, filename]
     return values[0], values[1], values[2]

# print(getPrediction('img1.jpeg'))
```

One limitation is that this model only can tell whether something is recyclable
or organic, it has no feature to distinguish trash.


arduino live
```arduino
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <L298NX2.h> //#include <L298N.h>
#include <IRremote.hpp>

int yled = 2;
int gled = 3;
int sy = 4;
int sg = 5;

int enA = 9;
int in1 = 8;
int in2 = 7;

//int enB = 11;
//int in3 = 10;
//int in4 = 12;
//int R_SIG = 10;
//int T_SIG = 11;
// Buffer for storing incoming serial data
String inputString = "";
boolean stringComplete = false;

#define IR_RECEIVE_PIN 6
//L298NX2 MOTORS(enA, in1, in2, enB, in3, in4);
L298N MOTORS(enA, in1, in2);
// Operation mode
int MODE = 3;        // Start in none mode by default
int SPEED = 100;
int MOVE_TIME = 20;
int PROD_TIME = 500;
void setup() {
  // put your setup code here, to run once:
  pinMode(yled, OUTPUT);
  pinMode(sy, INPUT);
  pinMode(gled, OUTPUT);
  pinMode(sg, INPUT);
  //pinMode(R_SIG, INPUT);
  //pinMode(T_SIG, INPUT);
  IrReceiver.begin(IR_RECEIVE_PIN, ENABLE_LED_FEEDBACK);
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect
  }

    // Reserve 200 bytes for the inputString
  inputString.reserve(200);
}
void loop3() {
  // Process completed commands
  if (stringComplete) {
    // Remove newline and carriage return characters
    inputString.trim();
    
    // Process the command
    processCommand(inputString);
    
    // Clear the string for new input
    inputString = "";
    stringComplete = false;
  }
  
  // Check for incoming serial data
  while (Serial.available()) {
    // Get the new byte
    char inChar = (char)Serial.read();
    
    // Add it to the inputString
    inputString += inChar; 
    
    // If the incoming character is a newline, set a flag
    // so the main loop can process the complete string
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}

// Function to process received commands
void processCommand(String command) {
  // Echo back the received command
  Serial.print("Received command: ");
  Serial.println(command);
  
  // Process specific commands
  if (command.equals("recycle")) {
    Serial.println("recycle ON");
    Serial.println("PI: FULL RECYCLE for 2 seconds");
    MOTORS.setSpeed(SPEED); // full speed
    MOTORS.forward();
    delay(PROD_TIME);
    MOTORS.stop();
    delay(PROD_TIME*3);
    MOTORS.backward();
    delay(PROD_TIME);
    MOTORS.stop();
  } 
  else if (command.equals("trash")) {
    Serial.println("trash on");
    Serial.println("PI: FULL TRASH for 2 seconds");
    MOTORS.setSpeed(SPEED); // full speed
    MOTORS.backward();
    delay(PROD_TIME);
    MOTORS.stop();
    delay(PROD_TIME*3);
    MOTORS.forward();
    delay(PROD_TIME);
    MOTORS.stop();
  }
  else if (command.equals("PING")) {
    Serial.println("PONG");
  }
  else {
    // For any other command, just acknowledge receipt
    Serial.println("Command not recognized. Try LED_ON, LED_OFF, or PING");
  }
  
  // Print a separator for readability
  Serial.println("-----");
}

void loop() {
  // put your main code here, to run repeatedly:
  int yval = digitalRead(sy);
  int gval = digitalRead(sg);
  //int rval = digitalRead(R_SIG);
  //int tval = digitalRead(T_SIG);
  int newMode = decideMode(yval, gval, 0);
  if (newMode != MODE) {
    Serial.print("Change in mode to ");
    Serial.println(newMode);
    MODE = newMode;
    return;
  }
  if (MODE == 1){
    if (stringComplete) {
      // Remove newline and carriage return characters
      inputString.trim();
      
      // Process the command
      processCommand(inputString);
      
      // Clear the string for new input
      inputString = "";
      stringComplete = false;
    }
    
    // Check for incoming serial data
    while (Serial.available()) {
      // Get the new byte
      char inChar = (char)Serial.read();
      
      // Add it to the inputString
      inputString += inChar; 
      
      // If the incoming character is a newline, set a flag
      // so the main loop can process the complete string
      if (inChar == '\n') {
        stringComplete = true;
      }
    }
    /*
    if (stringComplete){
      Serial.println("SECOND STRING COMPLETE");
      if (inputString.equals("recycle")){
        Serial.println("PI: FULL RECYCLE for 2 seconds");
        MOTORS.setSpeed(SPEED); // full speed
        MOTORS.forward();
        delay(PROD_TIME);
        MOTORS.stop();
      } else if (inputString.equals("trash")){
        Serial.println("PI: FULL TRASH for 2 seconds");
        MOTORS.setSpeed(SPEED); // full speed
        MOTORS.backward();
        delay(PROD_TIME);
        MOTORS.stop(); 
      }
      
    }
    */

  }

  if (IrReceiver.decode()) {
    uint8_t cmd = IrReceiver.decodedIRData.command;
    Serial.print("IR code received: ");
    Serial.println(cmd, HEX);

    if (cmd == 0x43) {
      Serial.println("IR Command: FULL FORWARD for 2 seconds");
      if (MODE == 2) {
        MOTORS.setSpeed(SPEED); // full speed
        MOTORS.forward();
        delay(MOVE_TIME);
        MOTORS.stop();
      }
    }
    else if (cmd == 0x44) {
      Serial.println("IR Command: FULL REVERSE for 2 seconds");
      if (MODE == 2) {
        MOTORS.setSpeed(SPEED); // full speed
        MOTORS.backward();
        delay(MOVE_TIME);
        MOTORS.stop();
      }
    }

    IrReceiver.resume();  // ready for next signal
  }
}

// 1 for recycle, 2 for trash, 3 for none, 4 for error
int inderence(int rval, int tval, int DEBUG){
  if (DEBUG){
    Serial.print("rval ");
    Serial.print(rval);
    Serial.print("   gval");
    Serial.print(tval);
    Serial.print(" 1,2,3 if statement:");
  }

  if(rval && !tval){
    return 1;
  }
  if(tval && !rval){
    return 2;
  } 
  if (!rval && !tval){
    return 3;
  }
  return 4;
}


int decideMode(int yval, int gval, int DEBUG){
  if (DEBUG){
    Serial.print("yval ");
    Serial.print(yval);
    Serial.print("   gval");
    Serial.print(gval);
    Serial.print(" 1,2,3 if statement:");
  }

  if(yval && !gval){ // add not the other
    digitalWrite(yled,HIGH);
    digitalWrite(gled, LOW);
    //Serial.print("1 ");
    return 1;
  }
  if(gval && !yval){
    digitalWrite(yled, LOW);
    digitalWrite(gled, HIGH);
    //Serial.print("2 ");
    return 2;
  } 
  if (!yval && !gval){
    digitalWrite(yled, LOW);
    digitalWrite(gled, LOW);
    //Serial.print("3 ");
    return 3;
  }
  digitalWrite(yled, LOW);
  digitalWrite(gled, LOW);
  return 4;
}

```
rpi4 

```python
#!/usr/bin/env python3
"""
live_trash_classifier_button.py
-------------------------------
OpenCV video preview + ImageNet inference on-demand:
press the BUTTON (GPIO 17) to classify the current frame,
press 'q' or Esc to quit.
Sends "recycle" or "trash" over serial to an attached Arduino.
"""

# --- std-lib ---
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# --- third-party ---
import cv2
import torch
import serial
from torchvision import transforms
from torchvision.models.inception import inception_v3, Inception_V3_Weights
from PIL import Image

# --- GPIO setup ---
import RPi.GPIO as GPIO

BUTTON_PIN = 17  # using BCM numbering; change if you like
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# --- Serial setup ---
SERIAL_PORT = "/dev/ttyUSB0"   # adjust as needed
BAUD_RATE    = 9600
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    # give Arduino a moment to reset
    time.sleep(5)
except serial.SerialException as e:
    GPIO.cleanup()
    sys.exit(f"Error: could not open serial port {SERIAL_PORT}: {e}")

# ---------- model & labels ----------
weights = Inception_V3_Weights.IMAGENET1K_V1
IMNET_LABELS = weights.meta["categories"]

model = inception_v3(weights=weights, aux_logits=True).eval().to("cpu")

# ---------- preprocessing ----------
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

RECYCLABLE = {
    'can','water bottle','pop bottle','soda bottle','wine bottle',
    'envelope','mailbag','postbag','menu','comic book','crossword puzzle',
    'cardigan','bath towel','wooden spoon','chain','cloak','bow',
    'ruler','suit','sunglasses','pajama','pyjama',"pj's",'jammies'
}

@torch.inference_mode()
def infer(pil_img: Image.Image, k: int) -> list[tuple[str, float]]:
    tensor = preprocess(pil_img).unsqueeze(0)           # (1,3,299,299)
    out    = model(tensor)

    # handle different return types
    if isinstance(out, (tuple, list)):
        logits = out[0]
    elif hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out
    logits = logits.squeeze(0)                          # (1000,)

    probs  = torch.softmax(logits, dim=0)
    top_p, top_i = probs.topk(k)
    return [(IMNET_LABELS[i], float(p)) for i, p in zip(top_i.tolist(), top_p.tolist())]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    ap.add_argument("--top-k", type=int, default=5, metavar="N",
                    help="How many predictions to print (default 5)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        GPIO.cleanup()
        sys.exit("Error: Could not open webcam")

    print(f"Press the BUTTON on GPIO {BUTTON_PIN} to classify current frame.")
    print("Press 'q' or Esc to quit.")

    prev_button = False
    latest_flag = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            cv2.imshow("Webcam", frame)

            # --- check for quit ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

            # --- poll button ---
            curr = GPIO.input(BUTTON_PIN)
            if curr and not prev_button:
                # simple debounce
                time.sleep(0.05)
                if GPIO.input(BUTTON_PIN):
                    # run inference
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    top_k = infer(pil_img, max(1, args.top_k))

                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{ts}] Top-{len(top_k)} predictions:")
                    for rank, (lbl, p) in enumerate(top_k, 1):
                        print(f"  #{rank:<2} {lbl:<25} {p*100:5.1f}%")

                    latest_flag = int(any(tok in top_k[0][0] for tok in RECYCLABLE))
                    if latest_flag:
                        decision = "recycle\n"
                    else:
                        decision = "trash\n"

                    # send decision over serial
                    ser.write(decision.encode('utf-8'))
                    ser.flush()
                    print(f"Sent over serial: {decision.strip()}")

            prev_button = curr

    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser.close()
        GPIO.cleanup()
        sys.exit(latest_flag)

if __name__ == "__main__":
    main()

```

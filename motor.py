import RPi.GPIO as GPIO          
from time import sleep
import requests

in1 = 16
in2 = 18
en = 22
temp1=1

GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en,GPIO.OUT)

GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)

p=GPIO.PWM(en,1000)
p.start(50)

API_URL = "http://127.0.0.1:8000/docs"

try:
    while True:
    # todo: test motors
        response = requests.get(API_URL)
        if response.status_code == 200:
            value = int(response.text.strip())
            print(f"API returned: {value}")
            
            if value == 1:
                print("Motor Forward")
                GPIO.output(in1, GPIO.HIGH)
                GPIO.output(in2, GPIO.LOW)
            elif value == 0:
                print("Motor Backward")
                GPIO.output(in1, GPIO.LOW)
                GPIO.output(in2, GPIO.HIGH)
            else:
                print("Unknown value from API")
        else:
            print(f"API error: {response.status_code}")
        
        time.sleep(2)  # Poll every 2 seconds

except KeyboardInterrupt:
    print("Cleaning up GPIO")
    GPIO.cleanup()

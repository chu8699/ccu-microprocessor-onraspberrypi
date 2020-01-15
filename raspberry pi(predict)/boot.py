import RPi.GPIO as GPIO
import subprocess
import signal
import os
import time
Button_pin = 4

def my_callback(channel):
    global process
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    process = subprocess.Popen("/home/pi/Documents/venv/speech/bin/python3 /home/pi/Documents/project/speech/predict.py"
                               ,shell = True, stdout=subprocess.PIPE, preexec_fn=os.setsid)
    
if __name__ == '__main__':
    # GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)       # Use BCM GPIO numbers
    GPIO.setup(Button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(Button_pin, GPIO.FALLING, callback=my_callback, bouncetime=300)
    process = subprocess.Popen("/home/pi/Documents/venv/speech/bin/python3 /home/pi/Documents/project/speech/predict.py"
                               ,shell = True, stdout=subprocess.PIPE, preexec_fn=os.setsid)
    while(1):
        time.sleep(1)
        continue
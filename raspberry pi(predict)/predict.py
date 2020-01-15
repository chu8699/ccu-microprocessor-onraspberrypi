# -*- coding: utf-8 -*-
"""
Created at 2019/12/8
@author: henk guo
"""
import RPi.GPIO as GPIO
import pyaudio
import struct
import wave
import matplotlib.pyplot as plt
import numpy as np
import twstock
from scipy.io import wavfile
import librosa
from tensorflow import keras
from tflite_runtime.interpreter import Interpreter
import librosa.display
import time
from gtts import gTTS
from pygame import mixer
import tempfile

# Define GPIO to LCD mapping
LCD_RS = 26
LCD_E  = 19
LCD_D4 = 13
LCD_D5 = 6
LCD_D6 = 5
LCD_D7 = 0
Button_pin = 2
 
# Define some device constants
LCD_WIDTH = 16    # Maximum characters per line
LCD_CHR = True
LCD_CMD = False
 
LCD_LINE_1 = 0x80 # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0 # LCD RAM address for the 2nd line
# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005

global button
BLOCKSIZE = 256
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
WIDTH = 2
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "tmp/output.wav"
LEN = 1 * RATE

 
def lcd_init():
    # Initialise display
    lcd_byte(0x33,LCD_CMD) # 110011 Initialise
    lcd_byte(0x32,LCD_CMD) # 110010 Initialise
    lcd_byte(0x06,LCD_CMD) # 000110 Cursor move direction
    lcd_byte(0x0C,LCD_CMD) # 001100 Display On,Cursor Off, Blink Off
    lcd_byte(0x28,LCD_CMD) # 101000 Data length, number of lines, font size
    lcd_byte(0x01,LCD_CMD) # 000001 Clear display
    time.sleep(E_DELAY)
 
def lcd_byte(bits, mode):
    # Send byte to data pins
    # bits = data
    # mode = True  for character
    #        False for command
     
    GPIO.output(LCD_RS, mode) # RS
     
    # High bits
    GPIO.output(LCD_D4, False)
    GPIO.output(LCD_D5, False)
    GPIO.output(LCD_D6, False)
    GPIO.output(LCD_D7, False)
    if bits&0x10==0x10:
        GPIO.output(LCD_D4, True)
    if bits&0x20==0x20:
        GPIO.output(LCD_D5, True)
    if bits&0x40==0x40:
        GPIO.output(LCD_D6, True)
    if bits&0x80==0x80:
        GPIO.output(LCD_D7, True)
     
    # Toggle 'Enable' pin
    lcd_toggle_enable()
 
    # Low bits
    GPIO.output(LCD_D4, False)
    GPIO.output(LCD_D5, False)
    GPIO.output(LCD_D6, False)
    GPIO.output(LCD_D7, False)
    if bits&0x01==0x01:
        GPIO.output(LCD_D4, True)
    if bits&0x02==0x02:
        GPIO.output(LCD_D5, True)
    if bits&0x04==0x04:
        GPIO.output(LCD_D6, True)
    if bits&0x08==0x08:
        GPIO.output(LCD_D7, True)
     
    # Toggle 'Enable' pin
    lcd_toggle_enable()
     
def lcd_toggle_enable():
    # Toggle enable
    time.sleep(E_DELAY)
    GPIO.output(LCD_E, True)
    time.sleep(E_PULSE)
    GPIO.output(LCD_E, False)
    time.sleep(E_DELAY)
 
def lcd_string(message,line):
    # Send string to display
 
    message = message.ljust(LCD_WIDTH," ")
 
    lcd_byte(line, LCD_CMD)
 
    for i in range(LCD_WIDTH):
        lcd_byte(ord(message[i]),LCD_CHR)
 

def is_silent(data, THRESHOLD):
    """Returns 'True' if below the threshold"""
    return max(data) < THRESHOLD


def extract_mfcc(file, fmax, nMel):
    y, sr = librosa.load(file)

    plt.figure(figsize=(3, 3), dpi=100)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nMel, fmax=fmax)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), fmax=fmax)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('/home/pi/Documents/project/speech/tmp/myimg/myImg.png', bbox_inches='tight', pad_inches=-0.1)

    plt.close()
    return

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def predict(interpreter):
    # MFCCs of the test audio
    top_k = 1
    extract_mfcc('/home/pi/Documents/project/speech/tmp/output.wav', 8000, 256)

    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                shear_range=0,
                                                                zoom_range=0,
                                                                horizontal_flip=False)
    test_generator = test_datagen.flow_from_directory('/home/pi/Documents/project/speech/tmp',
                                                      target_size=(250, 250),
                                                      batch_size=1,
                                                      class_mode='sparse')

    # Load the model
    Xts, _ = test_generator.next()

    # Predict the probability of each class
    set_input_tensor(interpreter, Xts)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    result = [(i, output[i]) for i in ordered[:top_k]]
    label_id, prob = result[0]
    print(f'{label_id}')
    return label_id

def getstockprice(stockid): 
    stock = twstock.realtime.get(stockid)
    name = stock['info']['fullname']
    price = stock['realtime']['latest_trade_price']
    return name, price;

def my_callback(channel):
    global button
    button = True
    print("yes i do")

def speak(sentence, lang):
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts=gTTS(text=sentence, lang=lang)
        tts.save('{}.mp3'.format(fp.name))
        mixer.init()
        mixer.music.load('{}.mp3'.format(fp.name))
        mixer.music.play(1)

if __name__ == '__main__':
    # GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)       # Use BCM GPIO numbers
    GPIO.setup(LCD_E, GPIO.OUT)  # E
    GPIO.setup(LCD_RS, GPIO.OUT) # RS
    GPIO.setup(LCD_D4, GPIO.OUT) # DB4
    GPIO.setup(LCD_D5, GPIO.OUT) # DB5
    GPIO.setup(LCD_D6, GPIO.OUT) # DB6
    GPIO.setup(LCD_D7, GPIO.OUT) # DB7
    GPIO.setup(Button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  
    # Initialise display
    lcd_init()
    lcd_string("program init",LCD_LINE_2)
    interpreter = Interpreter('/home/pi/Documents/project/speech/mfcc_cnn_model_all_tw.tflite')
    interpreter.allocate_tensors()
    
    button = False
    GPIO.add_event_detect(Button_pin, GPIO.FALLING, callback=my_callback, bouncetime=300) 
    
    while(1):
        stockid = ""
        lcd_string("push botton pls",LCD_LINE_2)
        if button == True:
            lcd_string("init microphone",LCD_LINE_2)
            for i in range(4):
                output_wf = wave.open('/home/pi/Documents/project/speech/tmp/output.wav', 'w')
                output_wf.setframerate(RATE)
                output_wf.setnchannels(CHANNELS)
                output_wf.setsampwidth(WIDTH)
                p = pyaudio.PyAudio()
                stream = p.open(format=p.get_format_from_width(WIDTH),
                                channels=CHANNELS,
                                rate=RATE,
                                input=True
                                )

                lcd_string("recording",LCD_LINE_2)
                start = False
                # Wait until voice detected

                while True:
                    input_string = stream.read(BLOCKSIZE, exception_on_overflow=False)
                    input_value = struct.unpack('h' * BLOCKSIZE, input_string)

                    silent = is_silent(input_value, 1300)
                    if not silent:
                        start = True

                    if start:
                        # Start recording
                        lcd_string("start",LCD_LINE_2)

                        nBLOCK = int(LEN / BLOCKSIZE)
                        numSilence = 0
                        for n in range(0, nBLOCK):

                            if is_silent(input_value, 800):
                                numSilence += 1

                            output_value = np.array(input_value)

                            if numSilence > RATE / 8000 * 7:
                                break

                            output_value = output_value.astype(int)
                            output_value = np.clip(output_value, -2 ** 15, 2 ** 15 - 1)

                            ouput_string = struct.pack('h' * BLOCKSIZE, *output_value)
                            output_wf.writeframes(ouput_string)

                            input_string = stream.read(BLOCKSIZE, exception_on_overflow=False)
                            input_value = struct.unpack('h' * BLOCKSIZE, input_string)

                        lcd_string("done",LCD_LINE_2)
                        start = False
                        num = predict(interpreter)
                        stockid += str(num)
                        
                        stream.stop_stream()
                        
                        stream.close()
                        p.terminate()
                        output_wf.close()
                        lcd_string(f"{stockid}",LCD_LINE_1)
                        break
            try:
                lcd_string("waiting",LCD_LINE_2)
                name, price = getstockprice(stockid)
                lcd_string(f"{stockid} : {price}",LCD_LINE_1)
                speak(f'{name}的股價為{price}', 'zh-tw')
            except:
                lcd_string("error",LCD_LINE_2)
                time.sleep(3)
            button = False
            

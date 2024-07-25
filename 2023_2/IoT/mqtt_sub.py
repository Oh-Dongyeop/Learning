import RPi.GPIO as GPIO
import paho.mqtt.client as paho
import time

GPIO.setmode(GPIO.BCM)  #BCM 모드 사용
GPIO.setwarnings(False)
GPIO.setup(18, GPIO.OUT)  # Pin number = 18, output

def toggle_led():
    GPIO.output(18, GPIO.HIGH)  # LED on
    time.sleep(1)  # 1 second delay
    GPIO.output(18, GPIO.LOW)  # LED off

def on_subscribe(client, userdata, mid, granted_qos):
    print(f"Subscribed: {mid} {granted_qos}")
def on_message(client, userdata, msg):
    toggle_led()
    print(f"{msg.topic} {msg.qos} {msg.payload}")
client = paho.Client()
client.on_subscribe = on_subscribe
client.on_message = on_message
client.connect("127.0.0.1", 1883)
client.subscribe("cbnu/#", qos=1)
# topic, 밴
client.loop_forever()

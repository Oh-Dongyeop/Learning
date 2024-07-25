import asyncio
import psutil
import RPi.GPIO as GPIO
import time
import paho.mqtt.client as paho

from iotc.models import Property,Command
from iotc import (
    IOTCConnectType,
    IOTCLogLevel,
    IOTCEvents,
    Command,
    CredentialsCache,
    Storage,
)
from iotc.aio import IoTCClient


TRIG = 23
ECHO = 24
GPIO.setmode(GPIO.BCM)  #BCM 모드 사용
GPIO.setwarnings(False)
GPIO.setup(18, GPIO.OUT)  # Pin number = 18, output
GPIO.setup(TRIG, GPIO.OUT) #전송핀 번호 지정 및 출력지정
GPIO.setup(ECHO, GPIO.IN)  #초음파 수신핀 번호 및 입력지정

def toggle_led():
    GPIO.output(18, GPIO.HIGH)  # LED on
    time.sleep(1)  # 1 second delay
    GPIO.output(18, GPIO.LOW)  # LED off

# MQTT
def on_connect(client, userdata, flags, rc):
    print(f"CONNACK received with code {rc}.")
def on_publish(client, userdata, mid):
    print(f"mid: {mid}")

client2 = paho.Client()
# client.username_pw_set('username', 'password')
client2.on_connect = on_connect
client2.on_publish = on_publish
client2.connect("127.0.0.1", 1883)
client2.loop_start()
count = 0

async def on_props(prop:Property):
    print(f"Received {prop.name}:{prop.value}")
    return True

async def on_commands(command: Command):
    client2.publish("cbnu/data", command.value)
    # if command.value == "LED ON":
    #     toggle_led()

    print("Received command {} with value {}".format(command.name, command.value))
    await command.reply()
# CPU 이용률
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.5)
    GPIO.output(TRIG, True) 	# 10us 펄스 전송
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    while GPIO.input(ECHO) == 0: # ECHO 시작시간 측정
        start = time.time()
    while GPIO.input(ECHO) == 1:
        stop = time.time()
    time_interval = stop - start    # 거리 계산
    distance = time_interval * 17000
    distance = round(distance, 2)
    return distance

scope_id = "0ne00A9C257"
device_id = "rpi-sensor"
key = "Br4z+aev1qAYe5BcaYOrOjSmwjFyLBzxPdUn4QSszho="
interface_id = "dtmi:iotNetworkSamplesDy:rpi_sensor_7ce;1"
client = IoTCClient(
    device_id,
    scope_id,
    IOTCConnectType.IOTC_CONNECT_DEVICE_KEY,
    key
    )
client.set_model_id(interface_id)
client.set_log_level(IOTCLogLevel.IOTC_LOGGING_ALL)
client.on(IOTCEvents.IOTC_PROPERTIES, on_props)
client.on(IOTCEvents.IOTC_COMMAND, on_commands)



async def main():
    await client.connect()
    await client.send_property({"writeableProp": 50})

    while not client.terminated():
        if client.is_connected():
            await client.send_telemetry({"cpu": get_cpu_usage()})
            await client.send_telemetry({"Distance": get_distance()})
        await asyncio.sleep(3)

asyncio.run(main())
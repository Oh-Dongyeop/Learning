import socket
import sys

UDP_IP = "127.0.0.1"
UDP_PORT = 5100

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

udp_sock.bind((UDP_IP, UDP_PORT))

print('Server is running.')
# client 에서 받은 메시지 출력 후 다시 보냄
while True:
    data , addr = udp_sock.recvfrom(1024)
    print("Received Data : {}".format(data.decode()))
    udp_sock.sendto(data,addr)
    if data.decode() == 'x' :
        break

udp_sock.close()
sys.exit()
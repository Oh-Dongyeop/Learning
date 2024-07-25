import socket
import sys

UDP_PORT = 5100

ad = input("IP address:")

udp_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

#키보드로 부터 문자열을 입력 받아서 서버로 전송
while True:
    data = input("Message: ")
    udp_socket.sendto(data.encode(),(ad,UDP_PORT))
    data , addr = udp_socket.recvfrom(1024)
    print ("recieved data :{}".format(data.decode()))
    if data.decode() == 'x':
        break

udp_socket.close()
sys.exit()
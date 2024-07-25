import socket
import sys

TCP_PORT = 5500

ad = input('IP address : ')

tcp_soket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
tcp_soket.connect((ad, TCP_PORT))

while True:
    data = input("message: ")
    if data.strip() == "":
        data = "Enter"
        
    tcp_soket.send(data.encode())
    data, addr = tcp_soket.recvfrom(1024)
    print ("recieved data :{}".format(data.decode()))
    if data.decode() == 'x':
        tcp_soket.close()
        break

sys.exit()

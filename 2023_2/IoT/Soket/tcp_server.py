import socket
import sys

TCP_IP = "127.0.0.1"
TCP_PORT = 5500

#소켓 생성
tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

tcp_sock.bind((TCP_IP,TCP_PORT))

tcp_sock.listen(1)

conn,addr = tcp_sock.accept()

print ("connect\n")

#client에서 온 데이터를 받아 출력 후 다시 보냄
while True:
    data = conn.recv(1024)
    msg = data.decode()
    print("Received Data : {}".format(data.decode()))
    conn.sendall(msg.encode())
    
    if msg == 'x' :
        tcp_sock.close()
        break

conn.close()
sys.exit()
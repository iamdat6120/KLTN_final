import socket 
import cv2 
import threading
import struct
import pickle
import time 




	

client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '192.168.1.6' # Here provide Drone IP 
port = 9999
client_socket.connect((host_ip,port))
data = b""
payload_size = struct.calcsize("Q")
while True:
    timen = time.time()
    while len(data) < payload_size:
        packet = client_socket.recv(1024) 
        if not packet: break
        data+=packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q",packed_msg_size)[0]
    while len(data) < msg_size:
        data += client_socket.recv(9000*1024)

    lst_inf_packed = data[:msg_size]
    data  = data[msg_size:]
    lst_inf = pickle.loads(lst_inf_packed)
    cv2.imshow("RECEIVING VIDEO FROM CACHE SERVER",lst_inf[0])
    key = cv2.waitKey(1) & 0xFF
    if key  == ord('q'):break




import socket, cv2, pickle, struct
import imutils
import cv2


server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_name  = socket.gethostname()
host_ip = socket.gethostbyname(host_name) # Enter the Drone IP address
print('HOST IP:',host_ip)
port = 9999
socket_address = (host_ip,port)
server_socket.bind(socket_address)
server_socket.listen()
print("Listening at",socket_address)



cftl=0
traffic_light = 'green'

i=0
client_socket,addr = server_socket.accept()
camera = False
if camera == True:
	vid = cv2.VideoCapture(0)
else:
	vid = cv2.VideoCapture(r'videotest\91.mp4')
try:
	tnow = 0
	print('CLIENT {} CONNECTED!'.format(addr))
	if client_socket:
		while(vid.isOpened()):
			cftl+=1
			if cftl%300==0:
				if traffic_light == 'green':
					traffic_light = 'red'

				else: 
					traffic_light = 'green'
			i+=1
			tnow+=1/30 #30 is fps of video
			_,frame = vid.read()
			if i==2:
				i=0
			# frame  = imutils.resize(frame,width=320)
				data = pickle.dumps([frame, tnow, traffic_light])
				message = struct.pack("Q",len(data))+data
				client_socket.sendall(message)

except Exception as e:
	print(f"CACHE SERVER {addr} DISCONNECTED")
	pass



# Read a video stream from camera(Frame by Frame)
import cv2

cap = cv2.VideoCapture(0) #this is to capture the device ID. if the argument is 0 then the device is from which it is executing. we can have multiple devices to capture the video

while True:
	ret,frame = cap.read() #cap.read returns 2 things 1st A Boolean value True for the video is being captured correctly if not then False, and the 2nd is the  frame  that has been captured
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if ret == False:
		continue
		
	cv2.imshow("Video Frame",frame)
	cv2.imshow("Gray Frame",gray_frame)
	#waitkey for user input -q then the loop will stop
	
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('a'):
		cv2.imshow("Captured in betweeen of the video",frame)
	
	
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break
		
cap.release()
cv2.destroyALLWindows()

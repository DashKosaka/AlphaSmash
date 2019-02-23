import cv2

def start():
	cap = cv2.VideoCapture(0)
	return cap

def finish(cap):
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

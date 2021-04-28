import imutils
import time
import cv2
import numpy as np

#Substitute the frame to the default template.png.
USE_WEBCAM = False

if(USE_WEBCAM == False):
    video_capture = cv2.VideoCapture("./90_0min_end5.avi") #input videofile

fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter("o_90_start_0to5_fps20.avi", fourcc, 20.0, (1260,930))
min_area = 1250
#Declaring the binary mask analyser object
#my_mask_analyser = BinaryMaskAnalyser()
#my_back_detector = BackProjectionColorDetector()
#my_back_detector.setTemplate(template) 
#Set the template 

while(True):

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if(frame is None): break #check for empty frames

       
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 199, 5)
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)

    thresh = cv2.bitwise_not(thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) != 0:
    
        for c in cnts:
		# if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
               continue
            c = max(cnts, key = cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            M = cv2.moments(c)
            cX = int(M['m10']/M['m00'])
            cY = int(M['m01']/M['m00'])
            cv2.circle(frame, (cX, cY), 1, (255, 0, 0), 2)
            print(cX,cY) #output_to_csv_file
    #Writing in the output file
    out.write(frame)

    #Showing the frame and waiting
    #for the exit command
    cv2.imshow('Original', frame) #show on window
    #cv2.imshow('Map', saliencyMap) #show on window
    if cv2.waitKey(1) & 0xFF == ord('q'): break #Exit when Q is pressed


#Release the camera
video_capture.release()
#print("Bye...")


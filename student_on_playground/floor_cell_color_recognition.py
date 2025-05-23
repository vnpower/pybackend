import cv2
import numpy as np
def empty(img):
    pass

video= cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,300)
cv2.createTrackbar("HUE Min","Trackbars",0,179,empty)
cv2.createTrackbar("HUE Max","Trackbars",179,179,empty)
cv2.createTrackbar("SAT Min","Trackbars",0,255,empty)
cv2.createTrackbar("SAT Max","Trackbars",255,255,empty)
cv2.createTrackbar("VAL Min","Trackbars",0,255,empty)
cv2.createTrackbar("VAL Max","Trackbars",255,255,empty)
cv2.createTrackbar("Area Min","Trackbars",0,30000,empty)



while True:
    ret,img=video.read()
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hue_min=cv2.getTrackbarPos("HUE Min","Trackbars")
    hue_max=cv2.getTrackbarPos("HUE Max","Trackbars")
    sat_min=cv2.getTrackbarPos("SAT Min","Trackbars")
    sat_max=cv2.getTrackbarPos("SAT Max","Trackbars")
    val_min=cv2.getTrackbarPos("VAL Min","Trackbars")
    val_max=cv2.getTrackbarPos("VAL Max","Trackbars")
    area_min=cv2.getTrackbarPos("Area Min","Trackbars")

    lower=np.array([hue_min,sat_min,val_min])
    upper=np.array([hue_max,sat_max,val_max])
    mask=cv2.inRange(hsv,lower,upper)
    # mask=cv2.inRange(hsv,(0,100,100),(10,255,255))

    contours, hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>300:
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            #cv2.putText(img,"Area: "+str(int(area)),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        if len(approx)==4:
            cv2.putText(img, "Points: "+str(len(approx)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow("Frame",img)
    cv2.imshow("HSV",hsv)
    cv2.imshow("Mask",mask)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
video.destroyAllWindows()
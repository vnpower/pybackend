import cv2

video= cv2.VideoCapture(0)

while True:
    ret,img=video.read()
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(0,100,100),(10,255,255))
    cv2.imshow("Frame",img)
    cv2.imshow("HSV",hsv)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
video.destroyAllWindows()
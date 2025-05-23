import cv2
import numpy as np

def detect_dominant_color(image, mask=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if mask is not None:
        hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
    dominant_hue = np.argmax(hist)
    return dominant_hue

def hue_to_color_name(hue):
    if hue < 10 or hue >= 160:
        return 'Red'
    elif hue < 25:
        return 'Orange'
    elif hue < 35:
        return 'Yellow'
    elif hue < 85:
        return 'Green'
    elif hue < 125:
        return 'Blue'
    elif hue < 160:
        return 'Purple'
    else:
        return 'Unknown'

def get_shape_name(approx):
    sides = len(approx)
    if sides == 3:
        return "Triangle"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif sides == 5:
        return "Pentagon"
    elif sides == 6:
        return "Hexagon"
    elif sides > 6:
        return "Circle"
    else:
        return "Unknown"

# Thay đổi đường dẫn tới file video của bạn
video_path = './floor_videos/floor_ver1.mp4'  # <- chỉnh lại tên file của bạn ở đây
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        if w * h < 1000:
            continue

        shape = get_shape_name(approx)

        # Mask để lấy màu trong hình
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        dominant_hue = detect_dominant_color(resized, mask)
        color_name = hue_to_color_name(dominant_hue)

        label = f'{shape} - {color_name}'
        cv2.drawContours(resized, [approx], -1, (0, 255, 0), 2)
        cv2.putText(resized, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Shape and Color Detection from Video", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import pyautogui
import cv2 

w, h = pyautogui.size()
x = w//2
y = h//2

#Get Webcam Feed
vid = cv2.VideoCapture(0)
ret = 1
#Initialise Yolo Model
model = YOLO('yolov8n.pt')
while ret:
    ret, frame = vid.read()
    #crosshair
    cv2.circle(frame, (x,y),5, (255,0,0),3)

    #Yolo Prediction
    results = model(frame)
    for result in results:
        for box in result.boxes.numpy():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(250,10,0),3)
            
    cv2.imshow("Webcam",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()






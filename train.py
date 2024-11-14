from ultralytics import YOLO
model = YOLO("/Users/madhuupadhyay/Documents/Stark_Industries/Heads-Up-Display/runs/detect/train10/weights/last.pt")
model.train(resume=True)
import cv2
img_file="car_Image.jpg"


classifier_file="car_detector.xml"


img=cv2.imread(img_file)

bnw=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

car_tracker=cv2.CascadeClassifier(classifier_file)

cars=car_tracker.detectMultiScale(bnw)
print(cars)

for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y),(x+w, y+h),(0,0,225),2)







cv2.imshow('My car Detector',img)




cv2.waitKey()


print("Code completed")
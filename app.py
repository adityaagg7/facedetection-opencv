import cv2
import os

face_Cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_Cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_Cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')


def detect(greyscale, colorframe):
    faces = face_Cascade.detectMultiScale(greyscale, 1.3, 7)
    for (x, y, w, h) in faces:
        cv2.rectangle(colorframe, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_grey = greyscale[y:y+h, x:x+w]
        roi_color = colorframe[y:y+h, x:x+w]
        eye = eye_Cascade.detectMultiScale(roi_grey, 1.7, 3)
        for (a, b, wi, hi) in eye:
            cv2.rectangle(roi_color, (a, b), (a+wi, b+hi), (0, 255, 0), 2)

        smile = smile_Cascade.detectMultiScale(roi_grey, 1.7, 22)
        for (a, b, wi, hi) in smile:
            cv2.rectangle(roi_color, (a, b), (a+wi, b+hi), (0, 255, 0), 2)

    return colorframe


key = input("Enter a for detection in image and b for detection using webcam:\n")

if key == 'a':
    path = os.path.join(os.getcwd(), 'facedetection-opencv',
                        'data', 'images', 'download.jpeg')
    print("Checking for:  ", path)
    cframe = cv2.imread(path)

    gframe = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)

    given = detect(gframe, cframe)
    cv2.imshow('cam', given)
    cv2.waitKey()
elif key == 'b':
    frame = cv2.VideoCapture(0)
    while 1:

        stat, cframe = frame.read()
        gframe = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)
        given = detect(gframe, cframe)
        cv2.imshow('cam', given)
        if cv2.waitKey(1) & 0xff == ord('q'):

            break

    frame.release()
    cv2.destroyAllWindows()
else:
    print("Wrong input")

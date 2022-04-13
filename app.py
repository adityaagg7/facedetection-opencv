import cv2


face_Cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_Cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')


def detect(greyscale, colorframe):
    faces = face_Cascade.detectMultiScale(greyscale, 1.3, 7)
    for (x, y, w, h) in faces:
        cv2.rectangle(colorframe, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_grey = greyscale[y:y+h, x:x+w]
        roi_color = colorframe[y:y+h, x:x+w]
        eye = eye_Cascade.detectMultiScale(roi_grey, 1.7, 7)
        for (a, b, wi, hi) in eye:
            cv2.rectangle(roi_color, (a, b), (a+wi, b+hi), (0, 255, 0), 2)

    return colorframe


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

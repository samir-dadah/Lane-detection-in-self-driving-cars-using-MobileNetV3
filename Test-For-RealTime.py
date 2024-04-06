import numpy as np
import cv2

cap = cv2.VideoCapture("/content/drive/MyDrive/car/Lane detect test data.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('/content/drive/MyDrive/car/fsd.mp4', fourcc, fps, (frame_width, frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)
        back, left, right = get_pred_for_mobilenet(learn.model,frame)[0]

        image=ld_detection_overlay(frame, left, right)

        output.write(image)

        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

output.release()

cv2.destroyAllWindows()

from queue import Queue
import cv2
from eye import Eye
from PyQt5.QtCore import pyqtSignal, QThread
import numpy as np
class VideoProcessing(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    def __init__(self, data_queue: Queue) -> None:
        self.data_queue = data_queue
        return super().__init__()
    def run(self) -> None:
        self.lock = False
        cap = cv2.VideoCapture(0)
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        face_cascade = cv2.CascadeClassifier('../../opencv_build/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('resources/haarcascade_eye.xml')
        
        FACTOR = 0.015

        self.starting_position = []

        while (cap.isOpened()):
            if self.lock:
                break
            ret, frame = cap.read()
            if ret == True:
                frame_height, frame_width = len(frame), len(frame[0])

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                directions = []
                for (x_face, y_face, width_face, height_face) in faces:
                    face_gray = gray[y_face : y_face + height_face, x_face : x_face + width_face]
                    face_color = frame[y_face : y_face + height_face, x_face : x_face + width_face]

                    detected_eyes = eye_cascade.detectMultiScale(face_gray)
                    detected_eyes = sorted(detected_eyes, key = lambda el:el[1])

                    for i, (eye_x, eye_y, eye_width, eye_height) in enumerate(detected_eyes[:2 if len(detected_eyes) > 2 else len(detected_eyes)]):
                        cv2.rectangle(face_color, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), (0, 255, 0), 2)
                        eye = Eye(face_gray[eye_y : eye_y + eye_height, eye_x : eye_x + eye_width], middle_block=(int(FACTOR * frame_width), int(FACTOR * frame_height)))
                        ey, ex = eye.get_center_of_frame()

                        cv2.circle(face_color, (ex + eye_x, ey + eye_y), 2, (255, 255, 0), thickness=-1)
                        if len(detected_eyes) < 2: 
                            continue
                        if len(self.starting_position) < 2:
                            self.starting_position.append((ex + eye_x + x_face, ey + eye_y + y_face))
                        else:
                            if abs(ex + eye_x + x_face - self.starting_position[0][0]) < abs(ex + eye_x + x_face - self.starting_position[1][0]):
                                eye.set_starting_position(self.starting_position[0])
                            else:
                                eye.set_starting_position(self.starting_position[1])
                            eye.set_eye_position((ex + eye_x + x_face, ey + eye_y + y_face))
                            directions.append(eye.get_direction())
                            cv2.circle(frame, (self.starting_position[i][0], self.starting_position[i][1]), 4, (255, 0, 0), thickness=-1)
                            
                self.change_pixmap_signal.emit(frame)
                self.data_queue.put(directions)


        self.data_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()
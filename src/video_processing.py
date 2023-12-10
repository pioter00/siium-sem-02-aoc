from queue import Queue
import cv2
from eye import Eye
from PyQt5.QtCore import pyqtSignal, QThread
import numpy as np
from time import sleep
import os


class VideoProcessing(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, data_queue: Queue) -> None:
        self.data_queue = data_queue
        return super().__init__()

    def run(self) -> None:
        self.lock = False
        sleep(5)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error opening video stream or file")

        face_cascade = cv2.CascadeClassifier('../resources/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('../resources/haarcascade_eye.xml')

        FACTOR = 0.015

        self.starting_position = []

        while cap.isOpened():
            if self.lock:
                break

            ret, frame = cap.read()

            if ret:
                frame_height, frame_width = len(frame), len(frame[0])

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                directions = []

                for (x_face, y_face, width_face, height_face) in faces:
                    face_gray = gray[y_face: y_face + height_face, x_face: x_face + width_face]
                    face_color = frame[y_face: y_face + height_face, x_face: x_face + width_face]

                    detected_eyes = eye_cascade.detectMultiScale(face_gray)
                    detected_eyes = sorted(detected_eyes, key=lambda el: el[1])

                    for i, (eye_x, eye_y, eye_width, eye_height) in enumerate(
                            detected_eyes[:2 if len(detected_eyes) > 2 else len(detected_eyes)]):

                        cv2.rectangle(face_color,
                                      (eye_x, eye_y),
                                      (eye_x + eye_width,
                                       eye_y + eye_height),
                                      (0, 255, 0),
                                      2)

                        eye = Eye(face_gray[eye_y: eye_y + eye_height, eye_x: eye_x + eye_width],
                                  middle_block=(int(FACTOR * frame_width), int(FACTOR * frame_height)))

                        ey, ex = eye.get_center_of_frame()

                        # Center of Hough eye
                        cv2.circle(face_color, (eye_x + ex, eye_y + ey), 2, (0, 255, 0), thickness=-1)

                        eye_blur = cv2.medianBlur(face_gray[eye_y: eye_y + eye_height, eye_x: eye_x + eye_width], 1)

                        circles = cv2.HoughCircles(eye_blur, cv2.HOUGH_GRADIENT, 1, 10,
                                                   param1=30,
                                                   param2=15,
                                                   minRadius=7,
                                                   maxRadius=15 )

                        if circles is not None:
                            circles = np.reshape(circles, (-1, 3))

                            for c in circles:
                                cv2.circle(face_color,
                                           (eye_x + int(c[0]), eye_y + int(c[1])),
                                           2,
                                           (255, 0, 0),
                                           thickness=-1)

                                cv2.circle(face_color,
                                           (eye_x + int(c[0]), eye_y + int(c[1])),
                                           int(c[2]),
                                           (255, 0, 0),
                                           thickness=1)

                            print(f'Possition: center {ex, ey} vs iris {circles[0][0], circles[0][1]}')

                        if len(detected_eyes) < 2:
                            continue
                        if len(self.starting_position) < 2:
                            self.starting_position.append((ex + eye_x + x_face, ey + eye_y + y_face))

                        else:
                            if abs(ex + eye_x + x_face - self.starting_position[0][0]) < abs(
                                    ex + eye_x + x_face - self.starting_position[1][0]):
                                eye.set_starting_position(self.starting_position[0])
                            else:
                                eye.set_starting_position(self.starting_position[1])

                            eye.set_eye_position((ex + eye_x + x_face, ey + eye_y + y_face))
                            directions.append(eye.get_direction())
                            # cv2.circle(frame, (self.starting_position[i][0], self.starting_position[i][1]), 4,
                            #            (255, 0, 0), thickness=-1)

                # TODO get status from here
                # if len(directions) == 2:
                #     if (directions[0].name == directions[1].name and
                #         directions[0].name != 'MIDDLE' and directions[1].name != 'MIDDLE'):
                #         print(f'Eyes looking {directions[0].name}')
                #     else:
                #         pass  # different data from one eye
                #
                # else:
                #     print('One eye blinked')

                # TODO after one successful message/reading set: self.starting_position = [] --> to calibrate once again

                self.change_pixmap_signal.emit(frame)
                self.data_queue.put(directions)

        self.data_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()

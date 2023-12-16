from queue import Queue
import cv2
from eye import Eye
from PyQt5.QtCore import pyqtSignal, QThread
import numpy as np
import math
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

        eye_status = {'Left': None,
                      'Right': None}  # Tracker for eye status

        # Iris detection parameters
        blur_mask_size = 7
        canny_param_1 = 30
        canny_param_2 = 15
        min_radius = 5
        max_radius = 12
        min_dist = 4

        # Iris position tracker
        iris_min_dist = 4

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

                    # TODO consider breaking if i >= 2 (more than 2 eye detected)
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

                        # Crop eye
                        eye_frame = face_gray[eye_y: eye_y + eye_height, eye_x: eye_x + eye_width]
                        # param_x = 0.3
                        # param_y = 0.15
                        # eye_frame = eye_frame[int(eye_frame.shape[0] * param_x / 2):
                        #                       eye_frame.shape[0] - int(eye_frame.shape[0] * param_x / 2),
                        #                       int(eye_frame.shape[1] * param_y / 2):
                        #                       eye_frame.shape[1] - int(eye_frame.shape[1] * param_y / 2)]

                        eye_blur = cv2.medianBlur(eye_frame, blur_mask_size)

                        circles = cv2.HoughCircles(eye_blur, cv2.HOUGH_GRADIENT, 1, min_dist,
                                                   param1=canny_param_1,
                                                   param2=canny_param_2,
                                                   minRadius=min_radius,
                                                   maxRadius=max_radius)

                        if circles is not None:
                            circles = np.reshape(circles, (-1, 3))

                            # Find circle, that is closest to middle of rectangle.
                            closest_circle = [0, 0, 0, 9999]  # x, y, r, dist
                            for c in circles:
                                dist = np.sqrt((ex - c[0])**2 + (ey - c[1])**2)

                                if dist < closest_circle[3]:
                                    closest_circle = [c[0], c[1], c[2], dist]

                            # Print closest circle.
                            cv2.circle(face_color,
                                       (eye_x + int(closest_circle[0]), eye_y + int(closest_circle[1])),
                                       2,
                                       (255, 0, 0),
                                       thickness=-1)
                            cv2.circle(face_color,
                                       (eye_x + int(closest_circle[0]), eye_y + int(closest_circle[1])),
                                       int(closest_circle[2]),
                                       (255, 0, 0),
                                       thickness=1)

                            # Set status of eye position tracker.
                            dist = closest_circle[3]
                            angle = math.degrees(math.atan2(-(closest_circle[1] - ey), closest_circle[0] - ex))
                            # print(f'Position eye {i} : dist {dist}, angle {angle}')

                            if i == 0:
                                eye_ = 'Left'
                            else:
                                eye_ = "Right"

                            if dist > iris_min_dist:
                                if angle > -45 and angle < 45:
                                    # this is mirror image, that's why left and right are switched
                                    eye_status[eye_] = 'Left'
                                elif angle > 45 and angle < 135:
                                    eye_status[eye_] = 'Up'
                                elif angle > 135 or angle < -135:
                                    eye_status[eye_] = 'Right'
                                elif angle > -135 and angle < -45:
                                    eye_status[eye_] = 'Down'

                            else:
                                eye_status[eye_] = 'Mid'

                        # if len(detected_eyes) < 2:
                        #     continue
                        # if len(self.starting_position) < 2:
                        #     self.starting_position.append((ex + eye_x + x_face, ey + eye_y + y_face))
                        #
                        # else:
                        #     if abs(ex + eye_x + x_face - self.starting_position[0][0]) < abs(
                        #             ex + eye_x + x_face - self.starting_position[1][0]):
                        #         eye.set_starting_position(self.starting_position[0])
                        #     else:
                        #         eye.set_starting_position(self.starting_position[1])
                        #
                        #     eye.set_eye_position((ex + eye_x + x_face, ey + eye_y + y_face))
                        #     directions.append(eye.get_direction())
                            # cv2.circle(frame, (self.starting_position[i][0], self.starting_position[i][1]), 4,
                            #            (255, 0, 0), thickness=-1)

                # TODO get status from here
                print(f'Eyes status {eye_status}')

                self.change_pixmap_signal.emit(frame)
                self.data_queue.put(directions)

        self.data_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()

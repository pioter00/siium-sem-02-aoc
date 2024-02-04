from queue import Queue
import cv2
from eye import Eye
from PyQt5.QtCore import pyqtSignal, QThread
import numpy as np
import math
from time import sleep

from src.consts import camera
from util import MyQueue
from chat_database import dialogues

class VideoProcessing(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, data_queue: Queue) -> None:
        self.data_queue = data_queue
        self.face_cascade = cv2.CascadeClassifier('../resources/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('../resources/haarcascade_eye.xml')
        # self.eye_status = {'Left': None,
        #                    'Right': None}  # Tracker for eye status

        self.blur_mask_size = 3
        self.canny_param_1 = 30
        self.canny_param_2 = 15
        self.min_radius = 3
        self.max_radius = 12
        self.min_dist = 2

        self.FACTOR = 0.015
        # Iris position tracker
        self.iris_min_dist = 4

        self.current_chat_dataset = 0

        return super().__init__()

    def detect_eyes_direction(self, ret, frame, eye_status_queue):

        eye_status = {'Left': None,
                      'Right': None}  # Tracker for eye status

        frame_height, frame_width = len(frame), len(frame[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        directions = []

        for (x_face, y_face, width_face, height_face) in faces:
            face_gray = gray[y_face: y_face + height_face, x_face: x_face + width_face]
            face_color = frame[y_face: y_face + height_face, x_face: x_face + width_face]

            detected_eyes = self.eye_cascade.detectMultiScale(face_gray)
            detected_eyes = sorted(detected_eyes, key=lambda el: el[1])
            if len(detected_eyes) < 2:
                print("none")

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
                          middle_block=(int(self.FACTOR * frame_width), int(self.FACTOR * frame_height)))

                ey, ex = eye.get_center_of_frame()

                # Center of Hough eye
                cv2.circle(face_color, (eye_x + ex, eye_y + ey), 2, (0, 255, 0), thickness=-1)

                # Crop eye
                eye_frame = face_gray[eye_y: eye_y + eye_height, eye_x: eye_x + eye_width]
                eye_blur = cv2.medianBlur(eye_frame, self.blur_mask_size)

                circles = cv2.HoughCircles(eye_blur, cv2.HOUGH_GRADIENT, 1, self.min_dist,
                                           param1=self.canny_param_1,
                                           param2=self.canny_param_2,
                                           minRadius=self.min_radius,
                                           maxRadius=self.max_radius)

                if circles is not None:
                    circles = np.reshape(circles, (-1, 3))

                    # Find circle, that is closest to middle of rectangle.
                    closest_circle = [0, 0, 0, 9999]  # x, y, r, dist
                    for c in circles:
                        dist = np.sqrt((ex - c[0]) ** 2 + (ey - c[1]) ** 2)

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

                    if dist > self.iris_min_dist:
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
        # TODO get status from here
        # print(f'Eyes status {eye_status}')
        eye_status_queue.push(eye_status)
        if eye_status_queue.isFull():
            left_eye_values = []
            right_eye_values = []
            for eye_status in eye_status_queue.list:
                left_eye_values.append(eye_status['Left'])
                right_eye_values.append(eye_status['Right'])

            left_eye_L = left_eye_values.count("Left")
            left_eye_R = left_eye_values.count("Right")
            right_eye_L = right_eye_values.count("Left")
            right_eye_R = right_eye_values.count("Right")

            if left_eye_L >= 6 and right_eye_L >= 6:
                return "LEFT"
            elif left_eye_R >= 6 and right_eye_R >= 6:
                return "RIGHT"
            else:
                return "MID"
        return "MID"

    def detect_eye_blink(self, ret, image) -> bool:
        # Convert the rgb image to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Applying bilateral filters to remove impurities
        gray = cv2.bilateralFilter(gray, 5, 1, 1)
        # to detect face
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # image = cv2.rectangle(image, (x, y), (x + w, y + h), (1, 190, 200), 2)
                # face detector
                roi_face = gray[y:y + h, x:x + w]
                # image
                roi_face_clr = image[y:y + h, x:x + w]
                # to detect eyes
                eyes = self.eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_face_clr, (ex, ey), (ex + ew, ey + eh), (255, 153, 255), 2)
                    if len(eyes) >= 2:
                        return False
                    else:
                        return True

    def print_chat_dataset(self, frame, width, height):
        currrent_dataset = dialogues[self.current_chat_dataset]
        width = int(width)
        height = int(height)
        padding = 70
        # up
        cv2.putText(frame, currrent_dataset[0], (int(width/2), padding), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        # down
        cv2.putText(frame, currrent_dataset[1], (int(width / 2), height - padding), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        # left
        cv2.putText(frame, currrent_dataset[2], (padding, int(height / 2)), cv2.FONT_HERSHEY_TRIPLEX, 1,(255, 255, 255), 2)
        # right
        cv2.putText(frame, currrent_dataset[3], (width - padding, int(height / 2)), cv2.FONT_HERSHEY_TRIPLEX, 1,(255, 255, 255), 2)

    def change_chat_dataset(self):
        datasets_count = len(dialogues)
        self.current_chat_dataset += 1
        if self.current_chat_dataset >= datasets_count:
            self.current_chat_dataset = 0

    def run(self) -> None:
        eye_status_queue = MyQueue(15)
        blink_status_queue = MyQueue(15)
        is_blinking = False

        self.lock = False
        # sleep(5)
        # cap = cv2.VideoCapture("../resources/video2.mp4")
        cap = cv2.VideoCapture(camera)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if not cap.isOpened():
            print("Error opening video stream or file")

        while cap.isOpened():
            if self.lock:
                break

            ret, frame = cap.read()

            if ret:
                blink = self.detect_eye_blink(ret, frame)
                direction = self.detect_eyes_direction(ret, frame, eye_status_queue)

                true_count = len([i for i in blink_status_queue.list if i == True])

                if not blink:
                    cv2.putText(frame, "Eye's Open", (70, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
                    blink_status_queue.push(False)
                    if is_blinking:
                        if (true_count/blink_status_queue.size) < 0.1:
                            print("Blink Detected.....!!!!")
                            is_blinking = False
                            self.change_chat_dataset()
                else:
                    cv2.putText(frame, "Eye's Close.....!!!!", (70, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
                    blink_status_queue.push(True)
                    is_blinking = True
                # print(direction)
                self.print_chat_dataset(frame, width, height)

                self.change_pixmap_signal.emit(frame)
                self.data_queue.put(None)

        self.data_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()

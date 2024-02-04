import sys
import cv2
import numpy as np

from queue import Queue
from time import sleep

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QApplication, QMainWindow, QWidget, QSlider
from PyQt5.QtGui import QImage, QPixmap

from scroll import Scroller
from video_processing import VideoProcessing
from consts import *

class WebCamThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        self.lock = False
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            if self.lock:
                break
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        cap.release()
        cv2.destroyAllWindows()

# class ChatThread(QThread):
#     update_chat_signal = pyqtSignal(str)
#
#     def update_chat(self, string: str):
#         self.update_chat_signal.emit(string)

class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.app = MainWidget()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Eye Tracking App')
        self.setCentralWidget(self.app)
        self.setGeometry(800, 400, 400, 200)
        self.show()

    def closeEvent(self, event) -> None:
        if type(self.app.thread) is not WebCamThread:
            self.app.eye_tracker.lock = True
        else:
            self.app.thread.lock = True
        sleep(0.5)
        return super().closeEvent(event)


class MainWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.dispaly_width = 1280
        self.display_height = 1024
        self.label = QLabel('Status:\n\nUnknown')

        self.start_acq_button = QPushButton('Start Eye Tracking')
        self.start_acq_button.clicked.connect(self.start_eye_tracking)

        self.stop_acq_button = QPushButton('Stop Eye Tracking')
        self.stop_acq_button.clicked.connect(self.stop_eye_tracking)
        self.stop_acq_button.setDisabled(True)

        self.calibrate_button = QPushButton('Calibrate')
        self.calibrate_button.clicked.connect(self.calibrate)
        self.calibrate_button.setDisabled(True)

        self.image_from_camera = QLabel(self)
        self.image_from_camera.resize(self.dispaly_width, self.display_height)

        self.thread = WebCamThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        self.data_queue = Queue()

        self.frames_to_wait = QLabel('Frames to wait: 10')
        self.frames_to_wait.setAlignment(Qt.AlignBottom)

        self.frames_slider = QSlider(Qt.Horizontal)
        self.frames_slider.setMinimum(5)
        self.frames_slider.setMaximum(60)
        self.frames_slider.setValue(10)
        self.frames_slider.valueChanged.connect(
            lambda: self.frames_to_wait.setText('Frames to wait: ' + str(self.frames_slider.value())))

        self.scroll_ticks = QLabel('Scroll ticks: 20')
        self.scroll_ticks.setAlignment(Qt.AlignBottom)

        self.ticks_slider = QSlider(Qt.Horizontal)
        self.ticks_slider.setMinimum(5)
        self.ticks_slider.setMaximum(40)
        self.ticks_slider.setValue(20)
        self.ticks_slider.valueChanged.connect(lambda: self.scroll_ticks.setText('Scroll ticks: ' + str(self.ticks_slider.value())))

        self.init_ui()

    def start_eye_tracking(self):
        self.thread.lock = True
        del self.thread
        sleep(0.5)
        self.start_acq_button.setDisabled(True)
        self.calibrate_button.setDisabled(False)
        self.stop_acq_button.setDisabled(False)
        self.frames_slider.setDisabled(True)
        self.ticks_slider.setDisabled(True)
        self.eye_tracker = VideoProcessing(self.data_queue)
        self.eye_tracker.change_pixmap_signal.connect(self.update_image)
        self.eye_tracker.update_chat_signal.connect(self.update_chat)
        self.scroller = Scroller(self.data_queue, self.frames_slider.value(), self.ticks_slider.value())
        self.scroller.start()
        self.eye_tracker.start()
        self.label.setText('Status:\n\nEye tracking is running')

    def stop_eye_tracking(self):
        self.eye_tracker.lock = True
        del self.eye_tracker
        del self.scroller
        sleep(0.5)
        self.thread = WebCamThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        self.stop_acq_button.setDisabled(True)
        self.calibrate_button.setDisabled(True)
        self.start_acq_button.setDisabled(False)
        self.frames_slider.setDisabled(False)
        self.ticks_slider.setDisabled(False)
        self.label.setText('Status:\n\nEye tracking is stopped\nCamera ready')

    def calibrate(self):
        self.eye_tracker.starting_position.clear()

    def init_ui(self):
        main_layout = QHBoxLayout()
        self.label.setAlignment(Qt.AlignTop)
        self.image_from_camera.setAlignment(Qt.AlignCenter)

        # Panel znajdujący się pod obrazem z kamery
        tracking_controls_panel = QHBoxLayout()

        # Przycisku w panelu pod obrazem z kamery
        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.start_acq_button)
        buttons_layout.addWidget(self.stop_acq_button)
        buttons_layout.addWidget(self.calibrate_button)

        tracking_controls_panel.addLayout(buttons_layout)
        tracking_controls_panel.addWidget(self.label)

        # Historia czatu po lewej stronie od widoku z kamery
        chat_history_panel = QWidget()
        chat_history_panel.setMinimumWidth(400)
        # chat_history_panel.setStyleSheet("background-color:red;")

        # Połączony widok z kamery oraz panelu z kontrolkami
        canera_with_controls_panel = QVBoxLayout()
        canera_with_controls_panel.addWidget(self.image_from_camera)
        canera_with_controls_panel.addStretch()
        canera_with_controls_panel.addLayout(tracking_controls_panel)

        # Połączenie panelu czatu razem z widokiem z kamery i kontrolkami
        main_layout.addWidget(chat_history_panel)
        main_layout.addStretch()
        main_layout.addLayout(canera_with_controls_panel)
        self.setLayout(main_layout)
        self.show()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_from_camera.setPixmap(qt_img)
        if self.label.text() == 'Status:\n\nUnknown':
            self.label.setText('Status:\n\nCamera ready')

    @pyqtSlot(str)
    def update_chat(self, value: str):
        print(f"CHAT UPDATED!: {value}")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.dispaly_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


def main():
    app = QApplication(sys.argv)
    ex = Main()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

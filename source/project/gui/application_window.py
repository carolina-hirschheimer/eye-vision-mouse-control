import numpy
from capturers import Capture
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget
from PyQt6.uic import loadUi

from frame_sources import FrameSource
from settings import settings


calibrate_top_left_bool = False
calibrate_top_right_bool = False
calibrate_bottom_left_bool = False
calibrate_bottom_right_bool = False

class Window(QMainWindow):

    # The following attributes are dynamically loaded from the .ui file
    startButton: QPushButton
    stopButton: QPushButton
    calibrateTopLeftButton: QPushButton
    calibrateTopRightButton: QPushButton
    calibrateBottomLeftButton: QPushButton
    calibrateBottomRightButton: QPushButton
    leftEyeThreshold: QSlider
    rightEyeThreshold: QSlider

    def __init__(self, video_source: FrameSource, capture: Capture):
        super(Window, self).__init__()
        loadUi(settings.GUI_FILE_PATH, self)
        with open(settings.STYLE_FILE_PATH, "r") as css:
            self.setStyleSheet(css.read())

        self.startButton.clicked.connect(self.start)
        self.stopButton.clicked.connect(self.stop)
        self.calibrateTopLeftButton.clicked.connect(self.calibrateTopLeft)
        self.calibrateTopRightButton.clicked.connect(self.calibrateTopRight)
        self.calibrateBottomLeftButton.clicked.connect(self.calibrateBottomLeft)
        self.calibrateBottomRightButton.clicked.connect(self.calibrateBottomRight)

        self.timer = None
        self.video_source = video_source
        self.capture = capture

    def start(self):
        self.video_source.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(settings.REFRESH_PERIOD)

    def stop(self):
        self.timer.stop()
        self.video_source.stop()

    def calibrateTopLeft(self):
        print("Calibrating Top Left")
        print(self.capture.keypoints[0].pt)
        with open('calibration_screen_cords.txt', 'r') as file:
            lines = file.readlines()
        lines[0] = "Top Left:{}\n".format(str(self.capture.keypoints[0].pt))
        with open('calibration_screen_cords.txt', 'w') as file:
            file.writelines(lines)

    def calibrateTopRight(self):
        print("Calibrating Top Right")
        print(self.capture.keypoints[0].pt)
        with open('calibration_screen_cords.txt', 'r') as file:
            lines = file.readlines()
        lines[1] = "Top Right:{}\n".format(str(self.capture.keypoints[0].pt))
        with open('calibration_screen_cords.txt', 'w') as file:
            file.writelines(lines)

    def calibrateBottomLeft(self):
        print("Calibrating Bottom Left")
        print(self.capture.keypoints[0].pt)
        with open('calibration_screen_cords.txt', 'r') as file:
            lines = file.readlines()
        lines[2] = "Bottom Left:{}\n".format(str(self.capture.keypoints[0].pt))
        with open('calibration_screen_cords.txt', 'w') as file:
            file.writelines(lines)

    def calibrateBottomRight(self):
        print("Calibrating Bottom Right")
        print(self.capture.keypoints[0].pt)
        with open('calibration_screen_cords.txt', 'r') as file:
            lines = file.readlines()
        lines[3] = "Bottom Right:{}\n".format(str(self.capture.keypoints[0].pt))
        with open('calibration_screen_cords.txt', 'w') as file:
            file.writelines(lines)

    def update_frame(self):
        frame = self.video_source.next_frame()
        face, l_eye, r_eye = self.capture.process(frame, self.leftEyeThreshold.value(), self.rightEyeThreshold.value())

        if face is not None:
            self.display_image(self.opencv_to_qt(frame))

        if l_eye is not None:
            self.display_image(self.opencv_to_qt(l_eye), window="leftEyeBox")

        if r_eye is not None:
            self.display_image(self.opencv_to_qt(r_eye), window="rightEyeBox")

    @staticmethod
    def opencv_to_qt(img) -> QImage:
        """
        Convert OpenCV image to PyQT image
        by changing format to RGB/RGBA from BGR
        """
        qformat = QImage.Format.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                qformat = QImage.Format.Format_RGBA8888
            else:  # RGB
                qformat = QImage.Format.Format_RGB888

        img = numpy.require(img, numpy.uint8, "C")
        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)  # BGR to RGB
        out_image = out_image.rgbSwapped()

        return out_image

    def display_image(self, img: QImage, window="baseImage"):
        """
        Display the image on a window - which is a label specified in the GUI .ui file
        """

        display_label: QLabel = getattr(self, window, None)
        if display_label is None:
            raise ValueError(f"No such display window in GUI: {window}")

        display_label.setPixmap(QPixmap.fromImage(img))
        display_label.setScaledContents(True)

    

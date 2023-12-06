import logging
from typing import Optional

import numpy
import cv2
from cv2.data import haarcascades
from settings import settings
import pyautogui

logger = logging.getLogger(__name__)
page_width = pyautogui.size().width
page_height = pyautogui.size().height
pyautogui.FAILSAFE = False

class CV2Error(Exception):
    pass


class HaarCascadeBlobCapture:
    """
    Class captures face and eyes using Haar Cascades.
    Detectes pupils using image processing with blob detection.
    Gaze estimation can be achieved by extracting x, y coordinates of the blobs
    Detailed description can be found here:
    https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
    """

    face_detector = cv2.CascadeClassifier(haarcascades + "haarcascade_frontalface_default.xml")
    eye_detector = cv2.CascadeClassifier(haarcascades + "haarcascade_eye.xml")
    blob_detector = None

    def __init__(self):
        self.previous_left_blob_area = 1
        self.previous_right_blob_area = 1
        self.previous_left_keypoints = None
        self.previous_right_keypoints = None
        self.keypoints = None
        self.eyes_detected = False
        self.one_blink_list = []
        self.two_blink_list = []

    def init_blob_detector(self):
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 1500
        self.blob_detector = cv2.SimpleBlobDetector_create(detector_params)

    def detect_face(self, img: numpy.ndarray) -> Optional[numpy.ndarray]:
        """
        Capture the biggest face on the frame, return it
        """

        coords = self.face_detector.detectMultiScale(img, 1.3, 5)

        if len(coords) > 1:
            biggest = (0, 0, 0, 0)
            for i in coords:
                if i[3] > biggest[3]:
                    biggest = i
            # noinspection PyUnboundLocalVariable
            biggest = numpy.array([i], numpy.int32)
        elif len(coords) == 1:
            biggest = coords
        else:
            return None

        for (x, y, w, h) in biggest:
            frame = img[y : y + h, x : x + w]
            return frame

    @staticmethod
    def _cut_eyebrows(img):
        """
        Primitively cut eyebrows out of an eye frame by simply cutting the top ~30% of the frame
        """
        if img is None:
            return img
        height, width = img.shape[:2]
        img = img[15:height, 0:width]  # cut eyebrows out (15 px)

        return img

    def detect_eyes(
        self, face_img: numpy.ndarray, cut_brows=True
    ) -> (Optional[numpy.ndarray], Optional[numpy.ndarray]):
        """
        Detect eyes, optionally cut the eyebrows out
        """
        coords = self.eye_detector.detectMultiScale(face_img, 1.3, 5)

        left_eye = right_eye = None

        if coords is None or len(coords) == 0:
            return left_eye, right_eye
        for (x, y, w, h) in coords:
            eye_center = int(float(x) + (float(w) / float(2)))
            if int(face_img.shape[0] * 0.1) < eye_center < int(face_img.shape[1] * 0.4):
                left_eye = face_img[y : y + h, x : x + w]
            elif int(face_img.shape[0] * 0.5) < eye_center < int(face_img.shape[1] * 0.9):
                right_eye = face_img[y : y + h, x : x + w]
            else:
                pass  # false positive - nostrill

            if cut_brows:
                return self._cut_eyebrows(left_eye), self._cut_eyebrows(right_eye)
            
            print("i am here")
            return left_eye, right_eye

    def blob_track(self, img, threshold, prev_area):
        _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, None, iterations=2)
        img = cv2.dilate(img, None, iterations=4)
        img = cv2.medianBlur(img, 5)
        self.keypoints = self.blob_detector.detect(img)
        if self.keypoints and len(self.keypoints) > 1:
            tmp = 1000
            for self.keypoint in self.keypoints:  # filter out odd blobs
                if abs(self.keypoint.size - prev_area) < tmp:
                    ans = self.keypoint
                    tmp = abs(self.keypoint.size - prev_area)

            self.keypoints = (ans,)
        return self.keypoints

    def draw(self, source, keypoints, dest=None):
        self.keypoints = keypoints
        try:
            aspect_ratio_threshold = 0.6  # Adjust this threshold based on your experimentation
            

            if self.keypoints == None:
                print("No eye detection", str(self.eyes_detected))

            else:
                print("YOU ARE LOOKING AT: ", self.keypoints[0].pt, self.eyes_detected)
                self.eyes_detected = True

                # Open the text file
                with open('calibration_screen_cords.txt', 'r') as file:
                    # Read lines from the file
                    lines = file.readlines()


                # Initialize variables to store coordinates
                top_left = None
                top_right = None
                bottom_left = None
                bottom_right = None

                # Iterate through lines and extract coordinates
                for line in lines:
                    # Split the line into parts using '(' and ')' as delimiters
                    parts = line.split('(')[1].split(')')[0].split(',')
                    # Extract x and y coordinates
                    x, y = map(float, parts)
                    
                    # Determine the position and assign coordinates accordingly
                    if 'Top Left' in line: 
                        top_left = (x, y)
                    elif 'Top Right' in line:
                        top_right = (x, y)
                    elif 'Bottom Left' in line:
                        bottom_left = (x, y)
                    elif 'Bottom Right' in line:
                        bottom_right = (x, y)

                top = min(top_left[1], top_right[1])
                bottom = max(bottom_right[1], bottom_left[1])
                right = max(top_right[0], bottom_right[0])
                left = min(top_left[0], bottom_left[0])

                #Proportion from coordinates of eye tracker to window coordinates
                ### (left, top) ---------- (0,0)
                ### (right, bottom) ------ (page_width, page_height)

                x_mouse = self.keypoints[0].pt[0] * page_width/right
                y_mouse = self.keypoints[0].pt[1] * page_height/bottom
                #pyautogui.moveTo(x_mouse,y_mouse,duration=0.3)

        except cv2.error as e:
            raise CV2Error(str(e))

    def debug_dump(self, frame):
        """
        Dump the frame to a folder for future debug
        """
        cv2.imwrite(str(settings.DEBUG_DUMP_LOCATION / f"{id(frame)}.png"), frame)

    def process(self, frame: numpy.ndarray, l_threshold, r_threshold):
        if not self.blob_detector:
            self.init_blob_detector()

        try:
            face = self.detect_face(frame)
            if face is None:
                return frame, None, None
            face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

            left_eye, right_eye = self.detect_eyes(face_gray)

            if left_eye is not None:
                left_key_points = self.blob_track(left_eye, l_threshold, self.previous_left_blob_area)

                kp = left_key_points or self.previous_left_keypoints
                left_eye = self.draw(left_eye, kp, frame)
                self.previous_left_keypoints = kp
            if right_eye is not None:
                right_key_points = self.blob_track(right_eye, r_threshold, self.previous_right_blob_area)

                kp = right_key_points or self.previous_right_keypoints
                right_eye = self.draw(right_eye, kp, frame)
                self.previous_right_keypoints = kp

            eye_detector = cv2.CascadeClassifier(haarcascades + "haarcascade_eye.xml")
            eyes = eye_detector.detectMultiScale(face,1.3,5,minSize=(50,50))

            if len(eyes) == 1 and self.eyes_detected == True:
                self.one_blink_list.append("BLINK")
                print("ONE BLINK")
            else:
                self.one_blink_list = []

            if len(eyes) == 0 and self.eyes_detected == True:
                self.two_blink_list.append("BLINK")
                print("TWO BLINK")

            else:
                self.two_blink_list = []

            if len(self.one_blink_list) > 3:
                pyautogui.click(pyautogui.position())
                print("LONG BLINK ==> LEFT CLICK")

            if len(self.two_blink_list) > 3:
                pyautogui.click(button='right', x=pyautogui.position()[0], y=pyautogui.position()[1])
                print("LONG BLINK ==> RIGHT CLICK")

            """if len(eyes) < 2 and self.eyes_detected == True:
                self.blink_list.append("BLINK")
                print("BLINK")

            else:
                self.blink_list = []

            if len(self.blink_list) > 3 and len(self.blink_list) < 10:
                pyautogui.click(pyautogui.position())
                print("LONG BLINK ==> LEFT CLICK")

            elif len(self.blink) >= 10:
                # left click 
                #print"""




            return frame, left_eye, right_eye
        except (cv2.error, CV2Error) as e:
            logger.error("error occurred: %s", str(e))
            logger.error(f"Thresholds: left: {l_threshold}, right: {r_threshold}")
            if settings.DEBUG_DUMP:
                self.debug_dump(frame)
            raise

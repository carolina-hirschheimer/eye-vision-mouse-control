import argparse
import sys

from capturers.haar_blob import HaarCascadeBlobCapture
from frame_sources import (CameraFrameSource, FileFrameSource,
                           FolderFrameSource, VideoFrameSource)
from gui.application_window import Window
from PyQt6.QtWidgets import QApplication

FRAME_SOURCES = {
    "camera": CameraFrameSource,
    "folder": FolderFrameSource,
    "file": FileFrameSource,
    "video": VideoFrameSource,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fs",
        "--frame-source",
        action="store",
        dest="frame_source",
        choices=FRAME_SOURCES.keys(),
        default="camera",
        help="What should be the frames source. A video/file, a folder with frames or a camera",
    )
    parser.add_argument(
        "-cam-id",
        "--camera-id",
        action="store",
        dest="camera_id",
        choices=FRAME_SOURCES.keys(),
        help="If your camera has an unusual ID in the system, pass it in this argument. Use only if your frame-source is camera(default)",
    )
    parser.add_argument(
        "running_status_arg",
        nargs="?",  # Make the argument optional
        default=None,  # Default value if not provided
        help="A custom argument for your logic",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    frame_source = FRAME_SOURCES[args.frame_source]

    frames_source_init_kwargs = {}
    if args.camera_id and args.frame_source == "camera":
        frames_source_init_kwargs["cam_id"] = args.camera_id

    
    if args.running_status_arg not in ["control_click", "control_mouse", "control_both", "calibrate"]:
        print("Invalid argument", args.running_status_arg)
        sys.exit()

    else:
        print("Running Status:", args.running_status_arg)
    capture = HaarCascadeBlobCapture(running_status = args.running_status_arg)

    

    app = QApplication(sys.argv)

    window = Window(frame_source(**frames_source_init_kwargs), capture)
    window.setWindowTitle("Eye Tracking")
    window.show()
    sys.exit(app.exec())

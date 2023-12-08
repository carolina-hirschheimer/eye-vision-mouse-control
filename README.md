# Eye Vision Mouse Control 
### Made by Carolina Hirschheimer and Rodrigo Nigri

## About the project

Mouseye is an innovative solution aimed at improving accessibility by enabling users to control their computer mouse through eye movements and blinks. This project facilitates interaction with computers for individuals with limited body mobility.

## Installation

### Linux or MacOS

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install poetry
$ poetry install
$ cd project
$ python main.py
```

### Windows
```bash
$ python3 -m venv venv
$ venv\Scripts\activate.bat
$ pip install poetry
$ poetry install
$ cd project
$ python main.py
```
## Configuration
After installing the required dependencies, the application can be executed in four different modes:

* **Calibration**: Used to calibrate the corners of the page. Look at one of the four corners of the computer window and click on the corresponding calibration button in the application interface. Argument: ```calibrate```.

* **Mouse Control**: Enables mouse control without clicks. Move the mouse by looking through the page at the desired location. Argument: ```control_mouse```.

* **Click Control**: Activates only the click control, and mouse control remains manual. Left-click by blinking with one eye and right-click by blinking with both eyes (hold the blink for a few seconds). Argument: ```control_click```.

* **Both Controls**: Enables both mouse and click controls. Argument: ```control_both```.

## Usage

To use the application , activate your virtual environment and run the following commands based on your operating system:

### Linux or MacOS

```bash
$ source venv/bin/activate
$ cd source/project/
$ python main.py [argument]
```

### Windows
```bash
$ venv\Scripts\activate.bat
$ cd source/project/
$ python main.py [argument]
```

Replace "[argument]" with one of the arguments mentioned in the Configuration section based on the functionality you want to use.

## Github Pages
Click here to see the [github pages](https://carolina-hirschheimer.github.io/eye-vision-mouse-control/) of this project. 

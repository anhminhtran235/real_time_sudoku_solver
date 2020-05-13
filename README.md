# Real Time Sudoku Solver

# 1/ Disclaimer:

- I knew almost nothing about Image Processing and Machine Learning when I started this project. Inspired by an awesome Augmented Reality Sudoku Solver made by geaxgx1 (Link below), I decided to make one of my own.

- Here is a list of sources I used to build this project:
    + geaxgx1's Youtube video: https://www.youtube.com/watch?v=QR66rMS_ZfA&t=130s . It contains amazing, abstract instructions on how to build a Real Time Sudoku Solver (Please notice he didn't upload any code)
    + Data used to train the CNN: Chars74K for Computer Fonts, instead of MNIST for hand written digits http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
    + OpenCV Tutorial Playlist (47 videos, I watched 75% of them): https://www.youtube.com/watch?v=kdLM6AOd2vc&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K
    + Sentdex's Deep Learning Playlist (11 videos): https://www.youtube.com/watch?v=wQ8BIBpya2k&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN
    + How to improve Digit Recognition Accuracy: https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
    + How to extract Sudoku Grid: https://maker.pro/raspberry-pi/tutorial/grid-detection-with-opencv-on-raspberry-pi
    + Interesting article in C++ about Sudoku Grabber: https://aishack.in/tutorials/sudoku-grabber-opencv-plot/
    + WarpPerspective: https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
    + Find 4 corners from contour: https://www.programcreek.com/python/example/89417/cv2.arcLength
    + Pixel viewers to see what goes wrong with my code: https://csfieldguide.org.nz/en/interactives/pixel-viewer/
    + And of course: https://stackoverflow.com/, https://docs.opencv.org/2.4/doc/tutorials/tutorials.html
- A lot of ideas (and codes) of this project came from others, as I'm very new to this field. But I still need to put A LOT of work on this project, and I'm very proud of what I have created :)
# 2/ How does it work?

- For a Demo and Explanation of how this application works, please watch my Youtube video: https://www.youtube.com/watch?v=uUtw6Syic6A
- You will need: Python 3, OpenCV 4, Tensorflow >= 2.1.0 and Keras.

# 3/ How can you run it?

- Just download all files, make sure you have the required installation of Python, OpenCV, Tensorflow and Keras, and run main.py
- You don't need to train CNN on your own. I have trained a CNN model and saved the architecture in digitRecoginition.h5
- If you want to try training the Convolution Network on your own computer, you will need to download Chars74K Dataset http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/, takes images 1-9 (We only need 1-9) and put them in folders "1", "2", ..., "9" respectively in the same directory with where you put all my Python files. After that, just run digitRecognition.py

April 2020

Created by Anh Minh Tran

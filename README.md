## DRIVER DROWSINESS DETECTION SYSTEM
Driver drowsiness detection is a car safety Technology which helps prevent accidents caused by the driver getting drowsy. The following code uses computer vision to observe the driver's face, either using a built-in camera or on mobile devices. It is responsible for 72,000 crashes, 44,000 injuries, and 800 deaths annually in the US. When driver fatigue or drowsiness is detected, the system alerts the driver to prevent possible accidents.

## WORKFLOW
If a face is found, we apply facial landmark detection and extract the eye regions. Now that we have the eye regions, we can compute the eye aspect ratio to determine if the eyes are closed. If the eye aspect ratio indicates that the eyes have been closed for a sufficiently long enough amount of time, we’ll sound an alarm to wake up the driver.

## ALGORITHM
Python’s dlib library uses Kazemi and Sullivan’s One Millisecond Face Alignment with an Ensemble of Regression Trees to implement the eye blink feature on the face.The program uses a facial training set to understand where certain points exist on facial structures. The program then plots the same points on region of interests in other images, if they exists. The program uses priors to estimate the probable distance between keypoints.

![image](https://github.com/user-attachments/assets/a85a795e-d8a9-4a42-963c-b2b11dffe2a9)

In Real Time Eye Blinking Using Facial Landmarks, Soukupová and Čech derive an equation that represents the Eye Aspect Ratio. The Eye Aspect Ratio is an estimate of the eye opening state. The algorithm is based on calculating an eye aspect ratio(EAR) which I will explain further. Each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye (as if you were looking at the person), and then working clockwise around the eye.It checks 20 consecutive frames and if the Eye Aspect ratio is less than 0.25, Alert is generated.
![eye1](https://github.com/user-attachments/assets/d7970be9-b91b-4a1f-8af5-cebdc4b49003)

“The Eye Aspect Ratio is a constant value when the eye is open, but rapidly falls to 0 when the eye is closed. It is calculated by the given formula.

![eye2](https://github.com/user-attachments/assets/69b4356c-3ed5-455c-afa2-ce3001489f4e)

## DEPENDENCIES
The example code is in Python (version 2.7 or higher will work).The `shape_predictor_68_face_landmarks.dat` is mentioned in the code. You can find it on `[Shape Predictor 68 features](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)` Extract the file in the project folder using ``bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2``

numpy==1.16.6 dlib==19.19.0 imutils==0.5.3 opencv_python==4.2.0.32 scipy==1.2.3 playsound==1.2.2
Use pip install -r requirements.txtto install the given requirements.

## USAGE
There are two files : `alarm_detection.py` and `drowsiness_detection.py` The former uses an alarm to detect a drowsy driver and the latter displays the alert on screen.

You can run the script using :

python drowsiness detection.py /for detection without a sound alert

python alarm_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav /for detection with a sound alert

## LIMITATIONS OF THE SYSTEM
According to available statistical data, over 1.3 million people die each year on the road and 20 to 50 million people suffer non-fatal injuries due to road accidents. Majority of these accidents happen due to a driver sleeping behind the wheel. A driver who falls asleep at the wheel loses control of the vehicle, an action which often results in a crash with either another vehicle or stationary objects. In order to prevent these devastating accidents, the state of drowsiness of the driver should be monitored. Although a drowsiness and a fatigue detection system is very crucial, yet it has its own limitations.

In the process of modeling, the researcher has to handle many challenging conditions that sometimes disturb and distract the process of gathering and analyzing data. For devices that have to capture the face and eye appearance, any condition that can cover face are undesirable. For instance, when the driver does not face towards the front, when lighting conditions are lacking, when driver wears accessories such as hats and glasses; the camera will not be able to capture the characteristics of the eyes and face.Beside camera, there are types of equipment attached to the face and other body regions to detect muscle contraction and eye activities (blinking, iris movement, etc).

The device usually cannot distinguish between natural/spontaneous expression or activity due to fatigue and intended (purposive) expression and movement. For instance, it is difficult to distinguish between yawning and talking activity, eyelid closure due to sleepiness or intended expression while talking or singing.

When the model is completely built and be used for the real monitoring system, the device used for monitoring is only a set of cameras. But when we are in the process of model building (to define appropriate formula and threshold to define alertness level), we have to use an intrusive device in the driver. The device usually is like a cap with wires and pads that should be patched on the face and head area. This device can be used to measure the eyes characteristics precisely instead of only counting on visual monitoring. This device also can disrupt image capturing process.

Last but not least challenges of these processes are individual differences. Everybody has his own characteristics, including an eye and face. For instance, there is a person with round eyes versus slanted eyes, long versus short eye lace, and also expressive versus nonexpressive person. These differences make model formulation more complicated.




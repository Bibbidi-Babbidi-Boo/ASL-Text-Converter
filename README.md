# ASL-Text-Converter

Here we use a deep learning model based on Yolo and Google Inception model v4, which are mixed and applied to the problem of American Sign Language Fingerspelling. To extract the image, background subtraction is applied, followed by morphological transformations, and the skin color from the face is used to localize the position of the hands in the image. After detecting the hands, they are passed into the neural network to detect the alphabet. The detection and prediction algorithm is run ten times, before declaring the alphabet which was detected the most number of times (within these ten trials). A two secon period is given for the user to change the sign befor starting the process again.

A video displaying the model working in a plain background, and the masking and morphological transforms used:
[Video for ASL Finger spelling in a clear background](https://drive.google.com/open?id=14OoIpv12VfPt8vS646wrP4tz7C4Ri2DE "Click to play")

A video showing the model working in a complex background:
[Video for ASL Finger spelling in a complex background](https://drive.google.com/open?id=1gHykSgf7ESIyXpDeAKlczY0QtzOsvlLH "Click to play")

The dataset used for the training is available here. The data is not owned by me, but found online.
https://drive.google.com/open?id=1SpY6TJGOjZw3_WH_4zeaoHmjBL5HMsul

Future Work:
Now only fingerspelling was acheived. It is now imperative to move on to words and sentences, where actions are used and hence the model must be changed from a CNN to a RNN. Apart from this, the background subtraction used is not optimal for the research work as it sometimes blurs or affects the foreground, and hence an RGBD camera or better algorithms are preferred

# glasses-detection
A simple deep learning python project I created when i was bored on a friday afternoon, it doesnt really serve any purpose but being a demo and/or project on my cv.

It uses OpenCV's CascadeClassifier to detect faces in the frames and a self-developed deep neural network to detect if the faces in the frame wear glasses.

For a sneak peek in to how the interface looks, head to the directory "gui" and take a look at the readme.md file.

## Repository structure
The code used to traing the deep neural network can be found in "DNN-training". More Information about the model can also be found there.

In "sample-in-and-output" an example pair of a possible input and the computed output can be found.

In "gui" the code for the user interface can be found and information on how to use it.

## Next steps
A few improvements to this program are in the works and will get finished when I find some free time:
1. A more lightweight neural network would help speed up the processing speed while not drastically decreasing the output quality.
2. As the current face detection algorithm isn't very stable in regard to rotations of the face, further face detection algorithms are an important further step. As annoted in the interface a Karhunen-Loeve transform and a Support Vector Machine based algorithm is in the works.
3. employing an optical flow algorithm to reduce redundant computations in subsequent frames.

# The user interface

Here is the code used for the final product, the interface where the user can select the video to be processed and see the video get processed.

## Interface
When running the interface an interface opens which allows to choose a ".mp4" file, which then gets processed (Note: The output will be named "filename_processed.mp4", where filename is the name of the original file (e.g. an input of "C:/testvideo.mp4" leads to an output of "C:/testvideo_processed.mp4").
![startview](https://user-images.githubusercontent.com/95757817/151229829-1bf36d4a-4615-41c0-8244-7dfecbbf0866.PNG)
After selecting a video, the interfaces allows the user to start the processing.
![selected](https://user-images.githubusercontent.com/95757817/151229828-f7438311-16dc-44f7-b8f6-aa7db8a53de3.PNG)

The second interface shows the progress of the processing and the latest processed image. 
![secondinterface](https://user-images.githubusercontent.com/95757817/151229820-b9ba7d41-2ea0-45a4-a95c-5b941494907c.PNG)
The third interface notifies the user that processing is finished.
![thirdinterface](https://user-images.githubusercontent.com/95757817/151229813-6d29bee6-9161-4ed1-bb3d-4f7d180f24af.PNG)
Output of the program is the input video with colored rectangles over the faces detected in the video. A green rectangle means that the neural network is 100% sure that the person wears glasses, yellow represents a 50% confidence and a red rectangle represents a 0% confidence (or a 100% confidence that the user doesnt wear glasses) values inbetween get interpolated using simple linear interpolation.


### example
Here is an example input and its finally computed output: [input](https://youtu.be/wzaVlkFZyY0), [output](https://youtu.be/5h32cv8SHQo)

## Usage
Download the [model here](https://www.kaggle.com/yvesbischofberger/glassdetectionmodel), unzip it and place it togheter with "cascade.xml" and the "videoProcessing.py" file in a folder, then run the "videoProcessing.py" file and follow the instructions on the screen. 
### videoProcessing_noNN.py
Alternatively use "videoProcessing_noNN.py" in place of "videoProcessing.py" it has the same functionalities minus the neural network to detect if the user wears glasses.

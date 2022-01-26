import numpy as np  # numerical methods
import tensorflow as tf # Neural Networks
import keras # Neural networks
from keras.applications import xception, inception_resnet_v2, resnet_v2 # models used for transfer learning
from tensorflow.keras.preprocessing import image_dataset_from_directory # data processing
from tensorflow.keras.layers import RandomFlip, RandomTranslation, RandomRotation, RandomZoom, RandomContrast, RandomCrop, Dense, Average, Flatten, Dropout # layers used in NN
from PIL import Image  # Image Functions
import os  # operating system, used for file path checks
import PySimpleGUIQt as sg  # PySimpleGui, used for user interface
import cv2  # opencv, used for video readout and write
import gc  # garbage collection


# predict faces
def face_predict(img_patch, model):
    #Convert img to RGB
    rgb = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
    #Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
    #Add dims to rgb_tensor (has dimension (1,2,3), needs (512,512,3)))
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    confidence = 1-(model.predict(rgb_tensor)[0][0])
    return confidence


# detect faces and detect if they wear glasses
def process_img(frame, model):
    # detects face(s) in an image, marks them and detects glasses
    # heavily inspired from https://github.com/shantnu/PyEng/blob/master/Image_Video/face_detect.py
    # convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # opencv cascadeClassifier
    face_cascade = cv2.CascadeClassifier("cascade.xml")
    # detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), maxSize=(512, 512))
    # detected faces
    for (x, y, w, h) in faces:
        # image patch of face for Deep neural network
        xmid = round((w / 2.0) + x)
        ymid = round((h / 2.0) + y)
        xlow = xmid - 256
        xhigh = xmid + 256
        ylow = ymid - 256
        yhigh = ymid + 256
        # get image patch
        image_patch = frame[ylow:yhigh, xlow:xhigh]
        shape = np.shape(image_patch)
        # padding in first dimension
        pad1 = 512-shape[0]
        left_pad1 = int(pad1/2.0)
        right_pad1 = pad1-left_pad1
        # padding in second dimension
        pad2 = 512-shape[1]
        left_pad2 = int(pad2/2.0)
        right_pad2 = pad2-left_pad2
        image_patch = np.pad(image_patch, ((left_pad1,right_pad1), (left_pad2,right_pad2),(0,0)), "reflect")
        # confidence of NN that this person is wearing glasses
        confidence = face_predict(image_patch, model)
        # mark faces in according color (linear interpolation, 100% sure glasses green, 50% sure yellow, 0% sure red)
        red = 255 if (confidence <= 0.5) else (255 - ((confidence - 0.5) * 2 * 255))
        green = 255 if (confidence >= 0.5) else (confidence * 2 * 255)
        blue = 0
        cv2.rectangle(frame, (x, y), (x + w, y + h), (blue, green, red), 2)
    # return processed image
    return frame


def main():
    # NEW CODE
    model = keras.models.load_model("../input/glassesmodels/model/model")
    # filepath of the video to be processed
    file_path = ""
    # create a GUI for the user
    sg.theme("Black")
    # describe the layout used when the user opens the application
    start_layout = [
        [sg.Text("Overlay Faces in video with colored boxes, depending on if they wear glasses or not")],
        # choose file
        [sg.Text("choose .mp4 file to be processed")],
        [sg.Input(size=(25, 1), key="file", enable_events=True), sg.FileBrowse(file_types=(("Video Files", "*.mp4"),))],
        [sg.Text("Face detection algorithm used"), sg.Radio("Haar Cascade", "method", default=True),
         sg.Radio("Karhunen-Loeve Transform (coming soon)", "method", disabled=True, text_color="grey"),
         sg.Radio("Support Vector Machine (coming soon)", "method", disabled=True, text_color="grey")],
        [sg.Button("Start processing video", key="process", disabled=True, button_color="grey")]
    ]
    # create the window containing the gui
    window = sg.Window("Face & Glasses Detection using Deep Learning", start_layout)
    while True:
        # read events and values
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "file":
            file_path = values["file"]
            # check if file_path is a .mp4 file
            if os.path.isfile(file_path):
                file_extension = os.path.splitext(file_path)[-1]
                if file_extension.lower() == ".mp4":
                    window["process"].update(button_color=("black", "white"))
                    window["process"].update(disabled=False)
        elif event == "process":
            # process video located at file_path
            video = cv2.VideoCapture(file_path)
            video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # describe the layout used when we're processing the video
            edit_layout = [
                [sg.Text("Progress:"),
                 sg.ProgressBar((video_length + 2), orientation="h", size=(500, 25), key="progressbar"),
                 sg.Text("", key="state")],
                [sg.Image(filename="", key="img")]
            ]
            window.close()
            editWindow = sg.Window("Face & Glasses Detection using Deep Learning", edit_layout)
            # this library for some reason needs to call editWindow.read before updating the window, for whatever reason
            event, values = editWindow.read(timeout=20)
            editWindow.maximize()
            # add _out to the filename and save it as out_path
            pathname, extension = os.path.splitext(file_path)
            path_split = pathname.split("/")
            path = "/".join(path_split[:-1])
            file_name = path_split[-1] + "_processed"
            out_path = path + "/" + file_name + extension
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            for i in range(video_length):
                ret, frame = video.read()
                # save image for easier handling
                cur = "img.jpg"
                cv2.imwrite(cur, frame)
                # process frame
                proc_frame = process_img(cur, model)
                # this library for some reason needs to call editWindow.read before updating the window, for whatever reason
                event, values = editWindow.read(timeout=20)
                # show latest processed frame
                imgbytes = cv2.imencode(".jpg", proc_frame)[1].tobytes()
                editWindow["img"].update(data=imgbytes)
                # update progress bar
                editWindow["progressbar"].update_bar(i + 1)
                editWindow["state"].update(f"{i + 1}/{video_length + 2} processed")
                # add processed frame to output
                out.write(proc_frame)
                # garbage collection
                gc.collect()
            out.release()
            video.release()
            editWindow.close()
            # describes the layout used when we're done processing the video
            done_layout = [
                [sg.Text(("Video Processing is finished, you can find your video at: " + out_path), key="donetext")],
                [sg.Text("Goodbye and have a nice day")]
            ]
            doneWindow = sg.Window("Video Processing is done", done_layout)
            while True:
                # read events and values
                event, _ = doneWindow.read(timeout=20)
                if event == "Exit" or event == sg.WIN_CLOSED:
                    break


main()

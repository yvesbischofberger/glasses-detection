# operating system, used for file path checks
import os
# PySimpleGui, used for user interface
import PySimpleGUIQt as sg
# opencv, used for video readout and write
import cv2
# garbage collection
import gc


# detect faces and detect if they wear glasses
def process_img(frame):
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # return processed image
    return frame


def main():
    # output to be accumulated
    output = 0
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
    # events (e.g. user changes facial recognition method)
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
                # process frame
                proc_frame = process_img(frame)
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

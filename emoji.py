import tkinter as tk
from tkinter import *
import cv2 
from PIL import Image, ImageTk
import os
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten   # imports layers from layers class from keras library
from keras.layers import Conv2D
# from keras.optimizers import Adam    
from keras.layers import MaxPooling2D
# from keras.preprocessing.image import ImageDataGenerator
import threading

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape = (48, 48, 1)))  
# con2D is keras convolution i.e filter-32
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))   
# relu-rectifier it is most used activation function
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))  # used to train subsets 
emotion_model.add(Dropout(0.25))  
emotion_model.add(Flatten())  # converts 2D to 1D
emotion_model.add(Dense(1024, activation='relu'))  
# dense -hidden layer, 1024 neurons
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))  # softmax used for o/p layer of neural n/w
emotion_model.load_weights('D:\Coding\emojify.miniproject\T.E. Mini-project\src\model.h5')
cv2.ocl.setUseOpenCL(False)  

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "fearful", 3: "Happy", 4: "Neutral", 5: " Sad ", 6: "Surprised"}

cur_path = os.path.dirname(os.path.abspath(__file__))    # __file__ is variable represents current path of dict 

emoji_dist = {0: cur_path+"/emojies/angry.png", 1: cur_path+"/emojies/disgust.png", 2: cur_path+"/emojies/fearful.png", 3: cur_path+"/emojies/happy.png", 4: cur_path+"/emojies/neutral.png", 5: cur_path+"/emojies/sad.png", 6: cur_path+"/emojies/surprised.png"}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
# numpy array of zeros with datatype
global cap1
show_text = [0]   # list of 1 element 0
global frame_number

# cap1 is variable to video
def show_subject():
    cap1 = cv2.VideoCapture(r"D:\Coding\emojify.miniproject\T.E. Mini-project\src\data\examples\WIN_20230510_09_54_49_Pro.mp4")  
    if not cap1.isOpened():
        print("Can't recognize the video")
    global frame_number
    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))  # to get total no of frames captured in video
    frame_number += 1
    if frame_number >= length:
        exit()
    cap1.set(1, frame_number)
    flag1, frame1 =cap1.read()   # reads nextframe in vd
    frame1 = cv2.resize(frame1,(600, 560))
    bounding_box = cv2.CascadeClassifier("C:\Python_3.9\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)   # apply face dect algo
    for(x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]  # extract region of intrest(roi) from detcted image
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)  # saves resized roi
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 225), 2, cv2.LINE_AA)
        show_text[0] = maxindex
    if flag1 is None:  # capturing major error or not
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()  #l-frame updates with c-frame
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        root.update()
        lmain.after(10, show_subject)  # continusly updating nd displaying frams with 10ms delay
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

def show_avatar():
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(frame2)  # converts image array to PIL image array
    imgtk2=ImageTk.PhotoImage(image=img2)   # creates tkinter compatibale image
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('arial', 45,'bold'))

    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(10, show_avatar)   # function called after 10ms

if __name__ == '__main__':   # main function
    frame_number = 0
    root=tk.Tk()   # main window is created
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2=tk.Label(master=root,bd=10)
    lmain3=tk.Label(master=root,bd=10,fg='#CDCDCD',bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=80,y=150)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=800,y=150)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")    # wxh+coordinates of screen
    root['bg']='black'
    exitButton = Button(root, text='Quit',fg="red", command=root.destroy, font=('arial',25,'bold')).pack(side=TOP,padx=10,pady=20)
    threading.Thread(target=show_subject).start()
    threading.Thread(target=show_avatar).start()
    root.mainloop()   # runs until GUI window is closed

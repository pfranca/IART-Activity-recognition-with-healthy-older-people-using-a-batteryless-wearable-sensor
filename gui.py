from tkinter import *
from tkinter import ttk
import os
import subprocess as sub
from multiprocessing import Process
from queue import Queue, Empty
import time


def runNN(): 
    global legenda
    legenda.destroy()
    legenda = Label(root,fg="red", text="Training Neural Network...")
    legenda.pack()
    root.update()
    text.delete('1.0', END)
    topFrame.update()
    os.system('python3 scriptNN.py > nn.txt')
    legenda.destroy()
    legenda = Label(root,fg="green", text="Neural Network Finnished, check test result above!")
    legenda.pack()
    root.update()
    processedFile = open('nn.txt', 'r')
    text.insert(END, processedFile.read())
    text.yview_pickplace("end")

def runKNN():
    global legenda
    legenda.destroy()
    legenda = Label(root,fg="red", text="Running K-Nearest Neighbours...")
    legenda.pack()
    root.update()
    text.delete('1.0', END)
    topFrame.update()
    os.system('python3 scriptKNN.py > knn.txt')
    legenda.destroy()
    legenda = Label(root,fg="green", text="K-Nearest Neighbours finnished, check results above!")
    legenda.pack()
    root.update()
    processedFile = open('knn.txt', 'r')
    text.insert(END, processedFile.read())

def runSVM():
    global legenda
    legenda.destroy()
    legenda = Label(root,fg="red", text="Running Suport Vector Machine... This may take a while")
    legenda.pack()
    root.update()
    text.delete('1.0', END)
    topFrame.update()
    os.system('python3 scriptSVM.py  > svm.txt')
    legenda.destroy()
    legenda = Label(root,fg="green", text=" Suport Vector Machine finnished, check results above!")
    legenda.pack()
    root.update()
    processedFile = open('svm.txt', 'r')
    text.insert(END, processedFile.read())


root = Tk();

root.title("Supervised Learning - Activity recognition");
root.geometry("1100x700")


legendaTopo = Label(root, text="G16 - Activity recognition with healthy older people using a batteryless wearable sensor Data Set ")
legendaTopo.pack(side=TOP)

topFrame = Frame(root);
topFrame.pack();
bottomFrame = Frame(root);
bottomFrame.pack(side=BOTTOM);


text = Text(topFrame, width=130,  height=30)
yscrollbar=Scrollbar(topFrame, orient=VERTICAL, command=text.yview)
yscrollbar.pack(side=RIGHT, fill=Y)
text["yscrollcommand"]=yscrollbar.set
text.pack(side=LEFT, fill=BOTH, expand = YES)

legenda = Label(root)

button1 = Button(bottomFrame, text="Neural Netorwrk", fg="red", padx=100,height= 3, command=runNN);
button2 = Button(bottomFrame, text="SVM", fg="green", padx=100,height= 3, command=runSVM);
button3 = Button(bottomFrame, text="KNN", fg="blue", padx=100,height= 3, command=runKNN);

button1.pack(side=LEFT);
button2.pack(side=LEFT);
button3.pack(side=LEFT);


root.mainloop();

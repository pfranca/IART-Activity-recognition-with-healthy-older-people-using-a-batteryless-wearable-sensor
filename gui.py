from tkinter import *

root = Tk();

root.title("Supervised Learning - Activity recognition");
root.geometry("1000x600")

topFrame = Frame(root);
topFrame.pack();
bottomFrame = Frame(root);
bottomFrame.pack(side=BOTTOM);

button1 = Button(bottomFrame, text="Neural Netorwrk");
button2 = Button(bottomFrame, text="SVM");
button3 = Button(bottomFrame, text="KNN");

button1.pack(side=LEFT);
button2.pack(side=LEFT);
button3.pack(side=LEFT);


root.mainloop();
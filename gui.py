# from tkinter import *
# import os
# import subprocess as sub
# from queue import Queue, Empty

# def run():
#     os.system('python3 test.py')

# root = Tk();

# root.title("Supervised Learning - Activity recognition");
# root.geometry("1000x600")

# topFrame = Frame(root);
# topFrame.pack();
# bottomFrame = Frame(root);
# bottomFrame.pack(side=BOTTOM);

# text = Text(topFrame)
# text.pack()

# button1 = Button(bottomFrame, text="Neural Netorwrk", command=run);
# button2 = Button(bottomFrame, text="SVM");
# button3 = Button(bottomFrame, text="KNN");

# button1.pack(side=LEFT);
# button2.pack(side=LEFT);
# button3.pack(side=LEFT);


# root.mainloop();

import sys
from itertools import islice
from subprocess import Popen, PIPE
from textwrap import dedent
from threading import Thread
import tkinter as tk # Python 3
from queue import Queue, Empty 

def iter_except(function, exception):
    """Works like builtin 2-argument `iter()`, but stops on `exception`."""
    try:
        while True:
            yield function()
    except exception:
        return

class DisplaySubprocessOutputDemo:
    def __init__(self, root):
        self.root = root

        # start dummy subprocess to generate some output
        self.process = Popen(["python3", "./test.py"], stdout=PIPE)

        # launch thread to read the subprocess output
        #   (put the subprocess output into the queue in a background thread,
        #    get output from the queue in the GUI thread.
        #    Output chain: process.readline -> queue -> label)
        q = Queue(maxsize=1024)  # limit output buffering (may stall subprocess)
        t = Thread(target=self.reader_thread, args=[q])
        t.daemon = True # close pipe if GUI process exits
        t.start()

        # show subprocess' stdout in GUI
        self.label = tk.Label(root, text="  ", font=(None, 12))
        self.label.pack(ipadx=4, padx=4, ipady=4, pady=4, fill='both')
        self.update(q) # start update loop

    def reader_thread(self, q):
        """Read subprocess output and put it into the queue."""
        try:
            with self.process.stdout as pipe:
                for line in iter(pipe.readline, b''):
                    q.put(line)
        finally:
            q.put(None)

    def update(self, q):
        """Update GUI with items from the queue."""
        for line in iter_except(q.get_nowait, Empty): # display all content
            if line is None:
                self.quit()
                return
            else:
                self.label['text'] = line # update GUI
                break # display no more than one line per 40 milliseconds
        self.root.after(40, self.update, q) # schedule next update

    # def quit(self):
    #     self.process.kill() # exit subprocess if GUI is closed (zombie!)
    #     self.root.destroy()


root = tk.Tk()
app = DisplaySubprocessOutputDemo(root)
#root.protocol("WM_DELETE_WINDOW", app.quit)
# center window
root.eval('tk::PlaceWindow %s center' % root.winfo_pathname(root.winfo_id()))
root.mainloop()

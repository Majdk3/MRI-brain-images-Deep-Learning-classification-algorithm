

import tkinter as tk
import displaying_all_the_results
from displaying_all_the_results import play
from tkinter import ttk  
from ttkthemes import ThemedTk
import os


window=tk.Tk()
greeting_meessage=tk.Label(text="Welcome  \n  \n Choose from below  a way to submit an image ",
                           foreground="black",
                           background="cyan",
                           width=60,
                           height=20,
                          font=("Courier", 18)
                          )
greeting_meessage.pack()

def manully_add_image(): 
    def get_results():
        val=entry1.get()
        val1=val.replace("\\","/")
        displaying_all_the_results.play(val1)      

    newWindow = tk.Toplevel(window) 
  

    newWindow.title("Submiting an image") 
  
 
    newWindow.geometry("800x600") 
  
    tk.Label(newWindow,  
          text ="please type the full dirctory of the image ").pack() 
    entry1 = tk.Entry(newWindow,fg="red", bg="white", width=50)
    entry1.pack()



    button = tk.Button(newWindow,
            text="Process the image",
            width=14,
            height=2,
            bg="grey",
            fg="cyan",
            command=get_results
            ).pack()
  

  
def open_list_images(): 
    text_varible = tk.StringVar()

    def get_results_list():
        val=entry.get()
        val1=val.replace("\\","/")
        label=val1[val1.find("=")+1]

        val1="Data set png/test/"+label+"/"+val1
        #print(val1)
        displaying_all_the_results.play(val1)
   
    def go(event): 
        cs = listbox_meningiomas.curselection() 
        text_varible.set(listbox_meningiomas.get(cs))

    def go1(event): 
        cs = listbox_gliomas.curselection() 
        text_varible.set(listbox_gliomas.get(cs))
 
    def go2(event): 
        cs = listbox_pituitary_tumors.curselection() 
        text_varible.set(listbox_pituitary_tumors.get(cs))
 
    newWindow1 = tk.Toplevel(window) 
  
    
    newWindow1.title("selecting an image ") 
  
    newWindow1.geometry("1024x1024") 
  
    tk.Label(newWindow1,  
          text ="double click on an image then press the process button").pack() 
    entry = tk.Entry(newWindow1,fg="red", bg="white", width=50,textvariable = text_varible)
    entry.pack()
    button = tk.Button(newWindow1,
                       text="Process the image",
            width=25,
            height=5,
            bg="grey",
            fg="cyan",
            command=get_results_list
            ).pack()

    listbox_meningiomas = tk.Listbox(newWindow1, height = 10,  
                  width = 60,  
                  bg = "grey", 
                  activestyle = 'dotbox',  
                  font = "Helvetica", 
                  fg = "yellow") 
  
  
    label = tk.Label(newWindow1, text = " test images for meningiomas")  
  
    for item,num in zip(os.listdir("Data set png/test/1"),range(1,len(os.listdir("Data set png/test/1"))+1)):
        listbox_meningiomas.insert(num, item) 
    listbox_meningiomas.bind('<Double-1>', go) 

    label.pack() 
    listbox_meningiomas.pack() 
    
    
    listbox_gliomas = tk.Listbox(newWindow1, height = 10,  
                  width = 60,  
                  bg = "grey", 
                  activestyle = 'dotbox',  
                  font = "Helvetica", 
                  fg = "yellow") 
  
  
    label = tk.Label(newWindow1, text = " test images for gliomas")  
  
    for item,num in zip(os.listdir("Data set png/test/2"),range(1,len(os.listdir("Data set png/test/2"))+1)):
        listbox_gliomas.insert(num, item) 
    listbox_gliomas.bind('<Double-1>', go1) 

    label.pack() 
    listbox_gliomas.pack()
    
    
    listbox_pituitary_tumors= tk.Listbox(newWindow1, height = 10,  
                  width = 60,  
                  bg = "grey", 
                  activestyle = 'dotbox',  
                  font = "Helvetica", 
                  fg = "yellow") 
  
  
    label = tk.Label(newWindow1, text = " test images for pituitary_tumors")  
  
    for item,num in zip(os.listdir("Data set png/test/3"),range(1,len(os.listdir("Data set png/test/3"))+1)):
        listbox_pituitary_tumors.insert(num, item) 
    listbox_pituitary_tumors.bind('<Double-1>', go2) 

    label.pack() 
    listbox_pituitary_tumors.pack()
    

  
  

btn = tk.Button(window,  
             text ="manually add an image",  
             command = manully_add_image) 
btn.pack() 
btn = tk.Button(window,  
             text ="pick an image  ",  
             command = open_list_images) 
btn.pack() 
  
# mainloop, runs infinitely 



#displaying the window
window.mainloop()

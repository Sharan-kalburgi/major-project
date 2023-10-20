from tkinter import *
def Train():
    """GUI"""
    import tkinter as tk
    import numpy as np
    import pandas as pd

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder

    root = tk.Tk()

    root.geometry("800x700+250+80")
    root.title("Check Heart Disease")
    root.configure(background="#8EB7FA")
    
    age = tk.IntVar()
    sex = tk.IntVar()
    chest_pain = tk.IntVar()
    restbp = tk.IntVar()
    chol = tk.IntVar()
    fbs = tk.IntVar()
    restecg = tk.IntVar()
    maxhr = tk.IntVar()
    exang = tk.IntVar()
    oldpeak = tk.DoubleVar()
    slope = tk.IntVar()
    ca = tk.IntVar()
    thal = tk.IntVar()
    
    #===================================================================================================================



    def Detect():
        e1=age.get()
        print(e1)
        e2=sex.get()
        print(e2)
        #b1=Lb1.get(Lb1.curselection())
        #e3.set(b1) 
        #value = Lb1.get(Lb1.curselection())
        #e3.set(value)  
        e3=chest_pain.get()
        print(e3)
        #print(type(e3))
        e4=restbp.get()
        print(e4)
        e5=chol.get()
        print(e5)
        e6=fbs.get()
        print(e6)
        e7=restecg.get()
        print(e7)
        e8=maxhr.get()
        print(e8)
        e9=exang.get()
        print(e9)
        e10=oldpeak.get()
        print(e10)
        e11=slope.get()
        print(e11)
        e12=ca.get()
        print(e12)
        e13=thal.get()
        print(e13)
        #########################################################################################
        
        from joblib import dump , load
        a1=load('HEART_DISEASE_MODEL.joblib')
        v= a1.predict([[e1, e2, e3, e4, e5, e6, e7, e8, e9,e10, e11, e12, e13]])
        print(v)
        if v[0]==1:
            print("Yes")
            yes = tk.Label(frame_alpr,text="Disease \nDetected!\nReport is Generated",background="red",foreground="white",font=('times', 20, ' bold '),width=15)
            yes.place(x=200,y=460)
            file = open(r"Report.txt", 'w')
            file.write("-----Patient Report-----\n As per input data and system model Heart Disease Detected for Respective Paptient."
                       "\n***Kindly Follow Medicatins***"
                    
                    )
            file.close()
            
        else:
            print("No")
            no = tk.Label(frame_alpr, text="No Disease \nDetected", background="green", foreground="white",font=('times', 20, ' bold '),width=15)
            no.place(x=200, y=460)
            file = open(r"Report.txt", 'w')
            file.write("-----Patient Report-----\n As per input data and system model No Heart Disease Detected for Respective Paptient."
                       "\n\n***Relax and Follow below mentioned Life Style to be Healthy as You Are!!!***"
                    
                    )
            file.close()

    frame_alpr = tk.LabelFrame(root, text=" --Display-- ", width=650, height=600, bd=5, font=('times', 15, ' bold '),bg="white")
    frame_alpr.grid(row=0, column=0, sticky='nw')
    frame_alpr.place(x=50, y=50)

    l1=tk.Label(frame_alpr,text="Age",font=('times', 20, ' bold '),width=10)
    l1.place(x=5,y=1)
    age=tk.Entry(frame_alpr,bd=2,width=5,font=("TkDefaultFont", 20),textvar=age)
    age.place(x=200,y=1)

    l2=tk.Label(frame_alpr,text="Sex",font=('times', 20, ' bold '),width=10)
    l2.place(x=5,y=50)
    sex=tk.Entry(frame_alpr,bd=2,width=5,font=("TkDefaultFont", 20),textvar=sex)
    sex.place(x=200,y=50)

    l3=tk.Label(frame_alpr,text="Chest Pain",font=('times', 20, ' bold '),width=10)
    l3.place(x=5,y=100)
    #chest_pain=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=chest_pain)
    #chest_pain.place(x=200,y=100)
    
    
    #Lb1 = Listbox(root,width=20,height=3)
    #Lb1.place(x=200,y=100)
    #Lb1.insert(1, "1")
    #Lb1.insert(2, "2")
    #Lb1.insert(3, "3")
    #chest_pain=Lb1.curselection()
    #Lb1.pack()
    R1 = Radiobutton(frame_alpr, text="Typical", variable=chest_pain, value=1).place(x=200,y=100)
    R2 = Radiobutton(frame_alpr, text="asymptomatic", variable=chest_pain, value=2).place(x=200,y=120)
    R3 = Radiobutton(frame_alpr, text="nontypical", variable=chest_pain, value=3).place(x=200,y=140)

    l4=tk.Label(frame_alpr,text="Trestbps",font=('times', 20, ' bold '),width=10)
    l4.place(x=5,y=170)
    restbp=tk.Entry(frame_alpr,bd=2,width=5,font=("TkDefaultFont", 20),textvar=restbp)
    restbp.place(x=200,y=170)

    l5=tk.Label(frame_alpr,text="Chol",font=('times', 20, ' bold '),width=10)
    l5.place(x=5,y=220)
    chol=tk.Entry(frame_alpr,bd=2,width=5,font=("TkDefaultFont", 20),textvar=chol)
    chol.place(x=200,y=220)

    l6=tk.Label(frame_alpr,text="FBS",font=('times', 20, ' bold '),width=10)
    l6.place(x=5,y=270)
    fbs=tk.Entry(frame_alpr,bd=2,width=5,font=("TkDefaultFont", 20),textvar=fbs)
    fbs.place(x=200,y=270)

    l7=tk.Label(frame_alpr,text="RestECG",font=('times', 20, ' bold '),width=10)
    l7.place(x=350,y=1)
    restecg=tk.Entry(frame_alpr,bd=2,width=5,font=("TkDefaultFont", 20),textvar=restecg)
    restecg.place(x=520,y=1)

    l8=tk.Label(frame_alpr,text="Thalach",font=('times', 20, ' bold '),width=10)
    l8.place(x=350,y=50)
    maxhr=tk.Entry(frame_alpr,bd=2,width=5,font=("TkDefaultFont", 20),textvar=maxhr)
    maxhr.place(x=520,y=50)

    l9=tk.Label(frame_alpr,text="ExANG",font=('times', 20, ' bold '),width=10)
    l9.place(x=350,y=120)
    exang=tk.Entry(frame_alpr,bd=2,width=5,font=("TkDefaultFont", 20),textvar=exang)
    exang.place(x=520,y=120)

    l10=tk.Label(frame_alpr,text="Old Peak",font=('times', 20, ' bold '),width=10)
    l10.place(x=350,y=170)
    oldpeak=tk.Entry(frame_alpr,bd=2,width=5,font=("TkDefaultFont", 20),textvar=oldpeak)
    oldpeak.place(x=520,y=170)

    l11=tk.Label(frame_alpr,text="Slope",font=('times', 20, ' bold '),width=10)
    l11.place(x=350,y=220)
    slope=tk.Entry(frame_alpr,bd=2,width=5,font=("TkDefaultFont", 20),textvar=slope)
    slope.place(x=520,y=220)

    l12=tk.Label(frame_alpr,text="Ca",font=('times', 20, ' bold '),width=10)
    l12.place(x=350,y=270)
    ca=tk.Entry(frame_alpr,bd=2,width=5,font=("TkDefaultFont", 20),textvar=ca)
    ca.place(x=520,y=270)

    l13=tk.Label(frame_alpr,text="Thal",font=('times', 20, ' bold '),width=10)
    l13.place(x=350,y=320)
    #thal=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=thal)
    #thal.place(x=200,y=600)
    R4 = Radiobutton(frame_alpr, text="Fixed", variable=thal, value=1).place(x=520,y=320)
    R5 = Radiobutton(frame_alpr, text="normal", variable=thal, value=2).place(x=520,y=350)
    R6 = Radiobutton(frame_alpr, text="reversable", variable=thal, value=3).place(x=520,y=380)

    button1 = tk.Button(frame_alpr,text="Submit",command=Detect,font=('times', 20, ' bold '),width=10, bg="blue",fg="white")
    button1.place(x=250,y=400)


    root.mainloop()

Train()
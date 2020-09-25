import csv
import time
import matplotlib.pyplot as plt
import pandas as pd 
import numpy
import csv
import os
import matplotlib as mpl
import numpy as np
from matplotlib import style
from mpl_toolkits.mplot3d import axes3d
from sklearn.model_selection import train_test_split
from datetime import datetime
from math import radians, sin, cos, acos, asin, sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from tkinter import *

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

#variables
itime = 0
ilat = 0 
ilong = 0
ialt = 0

ntime = 0
nlat = 0 
nlong = 0
nalt = 0

x_long = []
y_lat = []
z_al = []


# text = StringVar()
# text.set('old')
#path
csv_name = "D:\\SIH2020\\final_6th_sem\\new_405638_final.csv"
csv_data = "D:\\SIH2020\\final_6th_sem\\finaldata.csv"
classes = {1:'commercial aircraft',2:'drone'}

def datasep():
	global knn
	#data read and seperation
	# detectedout.set("data seperation started")
	print("data seperation started")
	dataset = pd.read_csv(csv_data)
	ds = pd.DataFrame(dataset,columns = ["catagory"])
	df = pd.DataFrame(dataset,columns = ["speed","alt"])

	Y = numpy.array([])
	for x in ds.index:
		Y = numpy.append (Y,[ds['catagory'][x]])

	X = df.to_numpy()

	X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=4)
	print("data seperation completed")
	# detectedout.set("data seperation completed")
	#Modal creation
	print("model creation started")
	# detectedout.set("model creation started")
	k_range = range(1,26)
	scores = {}
	scores_list = []
	for k in k_range:
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train,y_train)
		y_pred=knn.predict(X_test)
		scores[k] = metrics.accuracy_score(y_test,y_pred)
		scores_list.append(metrics.accuracy_score(y_test,y_pred))

	knn = KNeighborsClassifier(n_neighbors=5)
	knn.fit(X,Y)

	print("model creation completed")
	# detectedout.set("model creation completed")


def detection():
	#starting detection

	with open(csv_name) as f:
		reader = csv.reader(f)
		next(reader, None)
		int_row = next(reader)
		itime = datetime.fromtimestamp(int(int_row[0]))
		ilat = float(int_row[1])
		ilong = float(int_row[2])
		ialt = float(int_row[3])

		# print(ilong,ialt,ilat,itime)
		for r in reader:
			ntime=float(r[0])
			nlat=float(r[1])
			nlong=float(r[2])
			nalt=float(r[3])
			x_long.append(float(nlong))
			y_lat.append(float(nlat))
			z_al.append(float(nalt))
			ax1.plot(x_long,y_lat,z_al,marker='o',markersize=5)
			canvas.draw()
			ntime = datetime.fromtimestamp(int(ntime))

			if (((ntime-itime).total_seconds())%20 == 0):
				# print(ntime)
				dist = 6371.01 * (acos( (sin(ilat)*sin(nlat)) + ( cos(ilat)*cos(nlat)*cos(ilong - nlong) ) ) ) * 1000
				# print(dist)
				e_time = ((ntime - itime).total_seconds())
				# print(e_time)
				if dist != 0 and e_time != 0:
					s= dist/e_time
				
					# print(s)
					# altitude.append(alt)
					# speed.append(s)
					speed = s
					altitude = nalt


					ilat = nlat
					ilon = nlong
					ialt = nalt
					itime = ntime

					abp = np.column_stack((speed, altitude))
					# print(abp.shape) 	
					x_new = [[speed,altitude]]
					y_predict = int(knn.predict(abp))
					# print(int(y_predict))
					detectedout.set(classes[y_predict])
					# print(classes[y_predict]) 
					




def _quit():
	root.quit()     # stops mainloop
	root.destroy()


root = Tk()
root.wm_title("Telemetry clustering system")

detectedout = StringVar()
detectedout.set(None)

mpl.rcParams['toolbar'] = 'None' 
fig = Figure(figsize=(5, 4), dpi=100)
# t = np.arange(0, 3, .01)
# fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

# fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
# ax1.plot(x_long,y_lat,z_al,marker='o',markersize=5)
# ax1.set_title("np",fontsize=10,fontweight='bold')
ax1.set_xlabel('Longtitude',fontsize=10,fontweight='bold',labelpad=18)
ax1.set_ylabel('Latitude',fontsize=10,fontweight='bold',labelpad=18)
ax1.set_zlabel('Altitude',fontsize=10,fontweight='bold',labelpad=18)
ax1.ticklabel_format(useOffset=False)


canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


button = Button(master=root, text="Quit", command=_quit)
button.pack(side=BOTTOM,anchor = "sw",padx = 20)
button = Button(master=root, text="Initiate", command=datasep)
button.pack(side=BOTTOM,anchor = "sw" , padx=20)
root.update()
button = Button(master=root, text="start", command=lambda:detection())
button.pack(side=BOTTOM,anchor = "sw" ,padx=20)
det = Label(textvariable=detectedout ,bg="blue",fg="white",font=("comicsansms", 19 ,"bold"),padx= 30,pady =30)
det.pack(side=BOTTOM ,anchor= "se")
root.mainloop()
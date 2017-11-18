#	CS669 - Assignment 4 (Group-2) 
#	Last edit: 18/11/17
#	About: 
#		This program is a Classifier based on Batch-Perceptron Learning for linearly separable data.

import numpy as np
import math
import os
import sys
import matplotlib.pyplot as plt
import random
			
dimension=2									#	Dimension of data vectors.
l=1											#	Reduced dimension.
classes=[]									#	Contains data of the class.
classesName=[]
testClasses=[]								#	Contains test data.
G=[]										#	Stores the parameters of the discriminant function.

#	Program starts here...
print ("\nThis program is a Classifier based on Batch-Perceptron Learning for linearly separable data.\n")

#	Parsing Input... 
choice= raw_input("Do you want to use your own directory for features training/test input and output or default (o/d): ")

direct=""
directO=""
directT=""
choiceIn=1
choiceInner='a'

if choice=='o' or choice=='O':
	direct=raw_input("Enter the path (relative or complete) of the training feature data directory: ")
	directT=raw_input("Enter the path (relative or complete) of the test feature data directory: ")
	dimension=input("Enter the number of dimensions in the data (for input format, refer README): ")
	directO=raw_input("Enter the path (relative or complete) of the directory to store results of the classification: ")
elif choice=='d' or choice=='D':
	direct="../../../data/Input/Dataset 1/A/train"
	directT="../../../data/Input/Dataset 1/A/test"
	dimension=2
	directO="../../data/Output/test_results/"
else:
	print "Wrong input!. Exiting,"
	sys.exit()

minx,miny,maxx,maxy=0,0,0,0

for filename in os.listdir(direct):
	file=open(os.path.join(direct,filename))
	tempClassData=[]
	for line in file:
		number_strings=line.split()
		numbers=[float(num) for num in number_strings]
		if minx>numbers[0]:
			minx=numbers[0]
		if miny>numbers[1]:
			miny=numbers[1]
		if maxx<numbers[0]:
			maxx=numbers[0]
		if maxy<numbers[1]:
			maxy=numbers[1]
		tempClassData.append(numbers)
	classes.append(tempClassData)
	classesName.append(os.path.splitext(filename)[0])
	file.close()

for i in range(len(classesName)):
	for filename in os.listdir(directT):
		if os.path.splitext(filename)[0]==classesName[i]:
			file=open(os.path.join(directT,filename))
			tempClassData=[]
			for line in file:
				number_strings=line.split()
				numbers=[float(num) for num in number_strings]
				if minx>numbers[0]:
					minx=numbers[0]
				if miny>numbers[1]:
					miny=numbers[1]
				if maxx<numbers[0]:
					maxx=numbers[0]
				if maxy<numbers[1]:
					maxy=numbers[1]
				tempClassData.append(numbers)
			testClasses.append(tempClassData)
			file.close()

#	Calculating results for various values of eta(learning rate)...

colors=['r','b','g']
l=1
f=[]
AccuracyETA=[]
PrecisionETA=[]
RecallETA=[]
FMeasureETA=[]
ETA=[]
init=0.2
step=0.2

for q in range(5):

	eta=init+q*step
	print "Calculating results for eta = "+str(eta)+" ..."
	ETA.append(eta)
	for ci in range(len(classes)):
		for cj in range(ci+1,len(classes)):
			
			f.append(plt.figure(l))
			l+=1
			plt.subplot(111)
			D=[]
			for x in range(len(classes[ci])):
				p=(classes[ci][x],1)
				D.append(p)
			for x in range(len(classes[cj])):
				p=(classes[cj][x],-1)
				D.append(p)
			
			#	Initializing parameters of W and W0 and concatenating it as 'a'
			a=[1.0-random.random() for i in range(dimension+1)]
			
			#	Iterating till mis-classifications become 0.
			
			numMisClass=1
			
			while numMisClass:

				delA=[0 for i in range(dimension+1)]
				numMisClass=0

				for x in range(len(D)):
					zN=[]
					zN.append(1)
					for i in range(dimension):
						zN.append(D[x][0][i])
					aTzN=np.inner(np.array(a),np.array(zN))
					if (D[x][1]==1 and aTzN<0) or (D[x][1]==-1 and aTzN>=0):
						numMisClass+=1
						for i in range(len(zN)):
							delA[i]+=zN[i]*D[x][1]*eta
				for i in range(len(a)):
					a[i]+=delA[i]

			G.append(a)
			stepx=float(maxx-minx+2)/100
			stepy=float(maxy-miny+2)/100
			for i in range (100):
				for j in range (100):
					x=minx-1+stepx*i
					y=miny-1+stepy*j
					z=[1,x,y]
					az=np.inner(np.array(a),np.array(z))
					ind=0
					if az<0:
						ind=1
					plt.plot(x,y,'.',color=colors[ind])
			
			for x in range(len(D)):
				ind=0
				if D[x][1]==-1:
					ind=1
				plt.plot(D[x][0][0],D[x][0][1],'o',color=colors[(ind+1)%3])

			plt.xlabel('X->')
			plt.ylabel('Y->')
			f[l-2].suptitle("Decision Region plot (with training data) for class pair ("+str(ci+1)+","+str(cj+1)+") for eta = "+str(eta))
			f[l-2].savefig('../../data/Output/ClassPair_'+str(ci+1)+'_'+str(cj+1)+'_eta_'+str(eta)+'.png')

			del D

	f.append(plt.figure(l))
	l+=1
	plt.subplot(111)
	confusionMatrix=[[0 for i in range(len(classes))] for j in range(len(classes))]
	for i in range(len(testClasses)):
		for j in range(len(testClasses[i])):
			x=testClasses[i][j]
			z=[]
			z.append(1)
			for d in range(dimension):
				z.append(x[d])

			classLabel=i
			az=np.inner(np.array(G[0]),np.array(z))
			if az<0:
				az=np.inner(np.array(G[2]),np.array(z))
				if az<0:
					classLabel=2
				else:
					classLabel=1
			else:
				az=np.inner(np.array(G[1]),np.array(z))
				if az<0:
					classLabel=2
				else:
					classLabel=0
			
			confusionMatrix[classLabel][i]+=1

	stepx=float(maxx-minx+2)/100
	stepy=float(maxy-miny+2)/100
	for i in range (100):
		for j in range (100):
			x=minx-1+stepx*i
			y=miny-1+stepy*j
			z=[1,x,y]
			az=np.inner(np.array(G[0]),np.array(z))
			if az<0:
				az=np.inner(np.array(G[2]),np.array(z))
				if az<0:
					classLabel=2
				else:
					classLabel=1
			else:
				az=np.inner(np.array(G[1]),np.array(z))
				if az<0:
					classLabel=2
				else:
					classLabel=0
			plt.plot(x,y,'.',color=colors[classLabel])
	for i in range(len(testClasses)):
		for j in range(len(testClasses[i])):
			plt.plot(testClasses[i][j][0],testClasses[i][j][1],'o',color=colors[(i+1)%3])
	
	plt.xlabel('X->')
	plt.ylabel('Y->')
	f[l-2].suptitle("Decision Region plot (with test data) for all classes for eta = "+str(eta))
	f[l-2].savefig('../../data/Output/AllClasses_eta_'+str(eta)+'.png')

	Sumtot=0
	for i in range(len(classes)):
		for j in range(len(classes)):
			Sumtot+=confusionMatrix[i][j]

	confusionMatClass=[]
	for i in range(len(classes)):
		tempConfusionMatClass=[[0 for j in range(2)] for l in range(2)]
		sumin=0
		tempConfusionMatClass[0][0]=confusionMatrix[i][i]
		sumin+=tempConfusionMatClass[0][0]
		Sum=0
		for j in range(len(classes)):
			Sum+=confusionMatrix[i][j]
		tempConfusionMatClass[0][1]=Sum-tempConfusionMatClass[0][0]
		sumin+=tempConfusionMatClass[0][1]
		Sum=0
		for j in range(len(classes)):
			Sum+=confusionMatrix[j][i]
		tempConfusionMatClass[1][0]=Sum-tempConfusionMatClass[0][0]
		sumin+=tempConfusionMatClass[1][0]
		tempConfusionMatClass[1][1]=Sumtot-sumin
		confusionMatClass.append(tempConfusionMatClass)
	
	print "Data testing complete. Writing results in files for future reference."
	filer=open(os.path.join(directO,"results_"+str(eta)+".txt"),"w")

	filer.write("The Confusion Matrix of all classes together is: \n")
	for i in range(len(classes)):
		for j in range(len(classes)):
			filer.write(str(confusionMatrix[i][j])+" ")
		filer.write("\n")

	filer.write("\nThe Confusion Matrices for different classes are: \n")
	for i in range(len(confusionMatClass)):
		filer.write("\nClass "+str(i+1)+": \n")
		for x in range(2):
			for y in range(2):
				filer.write(str(confusionMatClass[i][x][y])+" ")
			filer.write("\n")

	Accuracy=[]
	Precision=[]
	Recall=[]
	FMeasure=[]

	filer.write("\nDifferent quantitative values are listed below.\n")
	for i in range(len(classes)):
		tp=confusionMatClass[i][0][0]
		fp=confusionMatClass[i][0][1]
		fn=confusionMatClass[i][1][0]
		tn=confusionMatClass[i][1][1]
		accuracy=float(tp+tn)/(tp+tn+fp+fn)
		if tp+fp:
			precision=float(tp)/(tp+fp)
		else:
			precision=-1.0
		if tp+fn:
			recall=float(tp)/(tp+fn)
		else:
			recall=-1.0
		if precision+recall:
			fMeasure=2*precision*recall/(precision+recall)
		else:
			fMeasure=-1.0
		filer.write("\nClassification Accuracy for class "+str(i+1)+" is "+str(accuracy)+"\n")
		if precision!=-1.0:
			filer.write("Precision for class "+str(i+1)+" is "+str(precision)+"\n")
		else:
			filer.write("Precision for class "+str(i+1)+" is -\n")
		if recall!=-1.0:
			filer.write("Recall for class "+str(i+1)+" is "+str(recall)+"\n")
		else:
			filer.write("Recall for class "+str(i+1)+" is -\n")
		if fMeasure!=-1.0:
			filer.write("F-measure for class "+str(i+1)+" is "+str(fMeasure)+"\n")
		else:
			filer.write("F-measure for class "+str(i+1)+" is -\n")
		Accuracy.append(accuracy),Precision.append(precision),Recall.append(recall),FMeasure.append(fMeasure)

	avgAccuracy,avgPrecision,avgRecall,avgFMeasure=0,0,0,0
	flagP,flagR,flagF=True,True,True
	for i in range (len(classes)):
		avgAccuracy+=Accuracy[i]
		if Precision[i]!=-1.0:
			avgPrecision+=Precision[i]
		else:
			flagP=False
		if Recall[i]!=-1.0:
			avgRecall+=Recall[i]
		else:
			flagR=False
		if FMeasure[i]!=-1.0:
			avgFMeasure+=FMeasure[i]
		else:
			flagF=False
	avgAccuracy/=len(classes)
	avgPrecision/=len(classes)
	avgRecall/=len(classes)
	avgFMeasure/=len(classes)

	filer.write("\nAverage classification Accuracy is "+str(avgAccuracy)+"\n")
	AccuracyETA.append(avgAccuracy)
	if flagP:
		filer.write("Average precision is "+str(avgPrecision)+"\n")
		PrecisionETA.append(avgPrecision)
	else:
		filer.write("Average precision is -\n")
		PrecisionETA.append(0)
	if flagR:
		filer.write("Average recall is "+str(avgRecall)+"\n")
		RecallETA.append(avgRecall)
	else:
		filer.write("Average recall is -\n")
		RecallETA.append(0)
	if flagF:
		filer.write("Average F-measure is "+str(avgFMeasure)+"\n")
		FMeasureETA.append(avgFMeasure)
	else:
		filer.write("Average F-Measure is -\n")
		FMeasureETA.append(0)
	filer.write("\n**End of results**")
	filer.close()
	del confusionMatClass
	del confusionMatrix

maxa=max(AccuracyETA)
maxaind=np.argmax(AccuracyETA)
maxp=max(PrecisionETA)
maxpind=np.argmax(PrecisionETA)
maxr=max(RecallETA)
maxrind=np.argmax(RecallETA)
maxf=max(FMeasureETA)
maxfind=np.argmax(FMeasureETA)

fig, ax = plt.subplots()
ax.plot(ETA, AccuracyETA, 'rs-', label='Average accuracy, max='+str("{0:.3f}".format(maxa))+' for eta='+str(ETA[maxaind]))
ax.plot(ETA, PrecisionETA, 'gs-', label='Average precision, max='+str("{0:.3f}".format(maxp))+' for eta='+str(ETA[maxpind]))
ax.plot(ETA, RecallETA, 'bs-', label='Average recall, max='+str("{0:.3f}".format(maxr))+' for eta='+str(ETA[maxrind]))
ax.plot(ETA, FMeasureETA, 'ys-', label='Average fmeasure, max='+str("{0:.3f}".format(maxf))+' for eta='+str(ETA[maxfind]))
plt.xlabel('Eta->')
plt.ylabel('Measures->')
leg = ax.legend(loc=4, bbox_to_anchor=(0.9, 1.0))
plt.savefig('../../data/Output/results.png', bbox_inches='tight')

#	End.
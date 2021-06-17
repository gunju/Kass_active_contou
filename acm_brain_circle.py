# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:04:11 2015

@author: gunjan_naik
"""
#import os
import cv2
import numpy as np
from scipy import signal,ndimage
from oct2py import octave,Oct2Py
import time
import matplotlib.pyplot as plt
try: 

    img=cv2.imread("circle.jpg",0)
    img=cv2.normalize(img.astype('float'),None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
    clone=img.copy()
    
    img=cv2.GaussianBlur(img,(3,3),np.ceil(3*1.0))

    ref_pt=[]
    
    cv2.namedWindow("img",cv2.WINDOW_AUTOSIZE)
    
    def click(event,x,y,flags,params):
        global ref_pt
        if event==cv2.EVENT_FLAG_RBUTTON:
            print('Contour initialization started')
            ref_pt=[(x, y)]
            cv2.setMouseCallback("img",click,img)
            cv2.circle(img,(int(x),int(y)),5,(0,0,255),1) 
            print(x,y)
            
        elif event==cv2.EVENT_FLAG_LBUTTON:
            print(x,y)
            cv2.circle(img,(int(x),int(y)),5,(0,0,255),1) 
            ref_pt.append((x, y))
    
    
    
    cv2.setMouseCallback("img",click)
    
    while 1:
        cv2.imshow("img", img)
        key = cv2.waitKey(10) #& 0xFF
        if key==27:
            break

    ref_pt.append(ref_pt[0])

    n=len(ref_pt)
    octave.push('n',n)
    octave.push('refpt', np.asarray(ref_pt))
    octave.eval("t=1:n;")
    octave.eval(" ts=1:0.1:n;")
    octave.eval("xys=spline(t,refpt,ts);")
    xys=octave.pull("xys")
    xs=xys[0,:,None]
    ys=xys[1,:,None]
    
    octave.push('xs',xs)
    octave.push('ys',ys)
    
    for i in range(len(xs)-1):
                cv2.line(img,(xs[i],ys[i]),(xs[i+1],ys[i+1]),(255,255,0),2)  
    
    cv2.imwrite("seed image.jpg",img)    
    cv2.imshow("img", img)
    cv2.waitKey(10)
    
    print('...Calculating Gradient map \n')
    
    # Parameters
    NoOfIter=5000                            #iterations
    octave.push('NoOfIter',NoOfIter)
    smth=clone#[:,:,0]                         #Smooth image

    alpha=3.0
    beta=0.6
    gamma=5.0

    kappa=0.2

    wl=0.0
    we=0.9
    wt=0.6
    
    [row,col]=smth.shape#Size of image
    
    tic=time.clock()
    
    #Computing external forces
    eline=smth
    [gradx,grady]=np.gradient(smth)
    
    eedge=-1*np.sqrt((np.multiply(gradx,gradx)+np.multiply(grady,grady)))
    
    # masks for taking various derivatives
    m1=[[-1,1]]
    m2=[[-1],[1]]
    m3=[[1,2,1]]
    m4=[[1,-2,1]]
    m5=[[1,-1],[-1,1]]
    
    
    
    cx=signal.convolve2d(smth,m1,mode='same')
    
    cy=signal.convolve2d(smth,m2,mode='same')
    
    cxx=signal.convolve2d(smth,m3,mode='same')
    
    cyy=signal.convolve2d(smth,m4,mode='same')
    
    cxy=signal.convolve2d(smth,m5,mode='same')
    
    #eeterm=[[0 for _ in range(row)]for _ in range(col)]
    eeterm=[]
    for i in range(0,row):
        for j in range(0,col):
            #eterm as defined in Kass snakes paper
            eeterm.append(((cyy[i,j]*cx[i,j]*cx[i,j]-2*cxy[i,j]*cx[i,j]*cy[i,j]+cxx[i,j]*cy[i,j]*cy[i,j]))/((1+cx[i,j]*cx[i,j]+cy[i,j]*cy[i,j])**1.5))
            
    eeterm=np.resize(eeterm,(row,col))
    
    eext=(wl*eline+we*eedge-wt*eeterm)

    [fx,fy]=np.gradient(eext)
    
    plt.figure(num='edge map')
    plt.imshow(eext,cmap='gray')
    plt.show

    [m,n]=xs.shape
    (mm,nn)=fx.shape
    
    #populating penta diagonal matrix
    b=[]
    
    b.append(beta)
    b.append(- (alpha + 4 * beta))
    b.append(2 * alpha + 6 * beta)
    b.append(b [1])
    b.append(b [0])
    
    octave.push('m',m)
    octave.push('b',b)
    octave.eval("A = b (1) * circshift (eye (m), 2);")
    octave.eval("A = A + b (2) * circshift (eye (m), 1);")
    octave.eval("A = A + b (3) * circshift (eye (m), 0);")
    octave.eval("A = A + b (4) * circshift (eye (m), - 1);")
    octave.eval("A = A + b (5) * circshift (eye (m), - 2);")
    
    #A=octave.pull('A')
    octave.push('gamma',gamma)
    octave.push('kappa',kappa)
    octave.push('fx',fx)
    octave.push('fy',fy)

    octave.eval("[L U]=lu(A+gamma.*eye(m,m));")
    octave.eval("Ainv=inv(U)*inv(L);")

    octave.addpath('C:/Users/gunjan_naik/Documents/GNU octave')
    octave.eval("[xs,ys]=interpolation(xs,ys,fx,fy,NoOfIter,Ainv,gamma,kappa);")
    xs_new=np.asarray(octave.pull("xs"))
    ys_new=np.asarray(octave.pull("ys"))

    pt=zip(xs_new,ys_new)
    while 1:
        cv2.imshow("im_new",clone)
        key = cv2.waitKey(1) & 0xFF
         #if cv2.waitKey(15)%0x100==27:
        for i in range(len(pt)-1):
            cv2.line(clone,pt[i],pt[i+1],(255,12,120),2)  
        cv2.line(clone,pt[len(pt)-1],pt[0],(255,12,120),2)  
            #break
        
        if key==27:
            break  
    cv2.imwrite("segemented_output_brain_tumor.jpg",clone)
    cv2.destroyAllWindows()
    toc=time.clock()
    elapsed=toc-tic
    print('Time elapsed '+str(elapsed))
    plt.figure()
    plt.imshow(clone)
    plt.show()
    

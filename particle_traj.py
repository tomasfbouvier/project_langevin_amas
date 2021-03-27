# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:45:13 2021

@author: To+
"""

import numpy as np
from simulacion_langevin import *


from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from Project import *


vx=0.1; vy=-0.3; vz= 0.1;x=0; y=1.5; z=0.

t=0
dt=0.001; xs=[];ys=[]; zs=[]; 
while y>0.55:
    x+=vx*dt; y+=vy*dt; z+=vz*dt
    
    t+=dt
    if(np.round(10*t,2)%5==0):
        xs.append(x); ys.append(y); zs.append(z)
        #plt.plot(x,z, 'r.')

#plt.plot(xs,zs)


x_pad=[];z_pad=[]
for i in range(len(xs)):
    print(i)
    aux=lange(xs[i],ys[i],zs[i],0,0,0, n_el=2E4)
    x_pad+=aux[0]; z_pad+=aux[1]
h,_,_,_=plt.hist2d(x_pad,z_pad, bins=1283, range=np.array([[0.,0.34990909090909095], [0., 0.34990909090909095]]))

    


im = np.flipud(h) 
# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
image_max = ndi.maximum_filter(im, size=20, mode='constant')

# Comparison between image_max and im to find the coordinates of local maxima
coordinates = peak_local_max(im, min_distance=50)

# display results
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
ax = axes.ravel()
"""
ax[0].imshow(im, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')
"""
ax[0].imshow(image_max, cmap=plt.cm.gray)
#ax[0].axis('off')
ax[0].set_title('Maximum filter')
ax[0].set_xlabel("X (m)")
ax[0].set_ylabel("Z (m)")

ax[1].imshow(im, cmap=plt.cm.gray)
ax[1].autoscale(False)
ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
#ax[1].axis('off')
ax[1].set_title('Peak local max')
ax[1].set_xlabel("X (m)")
ax[1].set_ylabel("Z (m)")

fig.tight_layout()

plt.show()
plt.figure()
points=[]; points_err=[]

models=models_generator(n_models=5, epochs=1)

for i in range(np.shape((coordinates))[0]):
    pad_region=im[coordinates[i,0]-55:coordinates[i,0]+55, coordinates[i,1]-55: coordinates[i,1]+55]
    predictions, pred_err=predictor(pad_region, models); 
    points.append([0.34990909090909095-(predictions[0,0]+coordinates[i,0]*0.34990909090909095/1283), predictions[0,1], predictions[0,2] +coordinates[i,1]*0.34990909090909095/1283])
    points_err.append([pred_err[0,0], pred_err[0,1], pred_err[0,2]])
points=np.array(points); points_err=np.array(points_err)
plt.plot(points[:,0], points[:,2], 'b.', label="predicted points")
plt.errorbar(points[:,0], points[:,2], xerr=points_err[:,0], yerr=points_err[:,2] , fmt = 'bx', capsize=3, alpha=.6 )

plt.plot(xs, zs, 'r.', label="real points")
plt.legend(loc="best")
plt.xlabel("x"); plt.ylabel("z")




plt.figure()
plt.plot(points[:,0], points[:,1], 'b.', label="predicted points")  
plt.errorbar(points[:,0], points[:,1], xerr=points_err[:,0], yerr=points_err[:,1] , fmt = 'bx', capsize=3, alpha=.6 )
  
plt.plot(xs, ys, 'r.',label="real points")
plt.xlabel("x"); plt.ylabel("y")
plt.legend(loc="best")
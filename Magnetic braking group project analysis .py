#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd 
from scipy.optimize import curve_fit
from scipy import optimize


# The code below reads in all the raw data sets as pandas dataframes 

# In[167]:


#Reads in the Aluminium data 
Al0 = pd.read_csv('Documents/Aluminium data\\B=0.csv')
Al1 = pd.read_csv('Documents/Aluminium data\\B=0.02.csv')
Al2 = pd.read_csv('Documents/Aluminium data\\B=0.04.csv')
Al3 = pd.read_csv('Documents/Aluminium data\\B=0.07.csv')
Al4 = pd.read_csv('Documents/Aluminium data\\B=0.08.csv')
Al5 = pd.read_csv('Documents/Aluminium data\\B=0.10.csv')
Al6 = pd.read_csv('Documents/Aluminium data\\B=0.14.csv')
Al7 = pd.read_csv('Documents/Aluminium data\\B=0.19.csv')
Al8 = pd.read_csv('Documents/Aluminium data\\B=0.24.csv')
Al9 = pd.read_csv('Documents/Aluminium data\\B=0.30.csv')

#Reads in the Brass data 
Br0 = pd.read_csv('Documents/Brass data\\B=0.csv')
Br1 = pd.read_csv('Documents/Brass data\\B=0.04.csv')
Br2 = pd.read_csv('Documents/Brass data\\B=0.05.csv')
Br3 = pd.read_csv('Documents/Brass data\\B=0.06.csv')
Br4 = pd.read_csv('Documents/Brass data\\B=0.08.csv')
Br5 = pd.read_csv('Documents/Brass data\\B=0.10.csv')
Br6 = pd.read_csv('Documents/Brass data\\B=0.15.csv')
Br7 = pd.read_csv('Documents/Brass data\\B=0.2.csv')
Br8 = pd.read_csv('Documents/Brass data\\B=0.25.csv')
Br9 = pd.read_csv('Documents/Brass data\\B=0.3.csv')

#Reads in the Copper data
Cu0 = pd.read_csv('Documents/Copper data\\B=0.csv')
Cu1 = pd.read_csv('Documents/Copper data\\B=0.04.csv')
Cu2 = pd.read_csv('Documents/Copper data\\B=0.06.csv')
Cu3 = pd.read_csv('Documents/Copper data\\B=0.08.csv')
Cu4 = pd.read_csv('Documents/Copper data\\B=0.11.csv')
Cu5 = pd.read_csv('Documents/Copper data\\B=0.15.csv')
Cu6 = pd.read_csv('Documents/Copper data\\B=0.2.csv')
Cu7 = pd.read_csv('Documents/Copper data\\B=0.25.csv')
Cu8 = pd.read_csv('Documents/Copper data\\B=0.29.csv')
Cu9 = pd.read_csv('Documents/Copper data\\B=0.32.csv')

#Reads in the Steel data
St0 = pd.read_csv('Documents/Steel data\\B=0.csv')
St1 = pd.read_csv('Documents/Steel data\\B=0.04.csv')
St2 = pd.read_csv('Documents/Steel data\\B=0.05.csv')
St3 = pd.read_csv('Documents/Steel data\\B=0.06.csv')
St4 = pd.read_csv('Documents/Steel data\\B=0.08.csv')
St5 = pd.read_csv('Documents/Steel data\\B=0.1.csv')
St6 = pd.read_csv('Documents/Steel data\\B=0.15.csv')
St7 = pd.read_csv('Documents/Steel data\\B=0.2.csv')
St8 = pd.read_csv('Documents/Steel data\\B=0.25.csv')
St9 = pd.read_csv('Documents/Steel data\\B=0.3.csv')


# In[168]:


#Sets the voltage and time columns to v and t
v = "Voltage U_A1 / V"
t = "Time t / s"


# The code below plots the raw data 

# In[169]:


plt.figure(figsize = (12, 6))
plt.yticks(fontsize=16,)
plt.xticks(fontsize=16,)
axis_font = {'fontname':'Arial', 'size':'20'}
plt.plot(Al3[t], Al3[v], color = 'black')
plt.xlabel('Time /s',**axis_font)
plt.ylabel('Voltage /V',**axis_font) 
plt.show()


# Below makes a very basic plot of period against time 

# In[21]:


#Sets diameter of magnet and radius if disk which remain constant  
D = 65e-3
R = 14.7e-2

#sigma is conductivity, d is thickness of disk, B is magnetic field, m is the mass 
def analysis(df, hv, lv, sigma, d, B, m):
    df = df.loc[(df[v] >= hv) | (df[v] <= lv)]
    df = df.round({v : 0})
    df['Diff'] = df[v].diff()
    df = df.loc[(df['Diff'] == -3.0) | (df['Diff'] == -2.0)]
    df['Period'] = df[t].diff()
    df = df.dropna()
    df['w'] = (2 * np.pi)/df['Period']
    return df


Al0 = analysis(Al0, 3.6, 0.7, 36.9e6, 1.07e-3, 0.0, 0.19675)
Al1 = analysis(Al1, 3.6, 0.7, 36.9e6, 1.07e-3, 23.7e-3, 0.19675)
Al2 = analysis(Al2, 3.6, 0.7, 36.9e6, 1.07e-3, 39.4e-3, 0.19675)
Al3 = analysis(Al3, 3.6, 0.7, 36.9e6, 1.07e-3, 66.6e-3, 0.19675)
Al4 = analysis(Al4, 3.6, 0.7, 36.9e6, 1.07e-3, 82.8e-3, 0.19675)
Al5 = analysis(Al5, 3.6, 0.7, 36.9e6, 1.07e-3, 102.5e-3, 0.19675)
Al6 = analysis(Al6, 3.6, 0.7, 36.9e6, 1.07e-3, 136.3e-3, 0.19675)
Al7 = analysis(Al7, 3.6, 0.7, 36.9e6, 1.07e-3, 192.8e-3, 0.19675)
Al8 = analysis(Al8, 3.6, 0.7, 36.9e6, 1.07e-3, 244.0e-3, 0.19675)
Al9 = analysis(Al9, 3.6, 0.7, 36.9e6, 1.07e-3, 297.0e-3, 0.19675)

Br0 = analysis(Br0, 3.4, 0.7, 15.9e6, 0.86e-3, 0.0, 0.53218)
Br1 = analysis(Br1, 3.4, 0.7, 15.9e6, 0.86e-3, 40.1e-3, 0.53218)
Br2 = analysis(Br2, 3.4, 0.7, 15.9e6, 0.86e-3, 49.3e-3, 0.53218)
Br3 = analysis(Br3, 3.4, 0.7, 15.9e6, 0.86e-3, 57.8e-3, 0.53218)
Br4 = analysis(Br4, 3.4, 0.7, 15.9e6, 0.86e-3, 80.9e-3, 0.53218)
Br5 = analysis(Br5, 3.4, 0.7, 15.9e6, 0.86e-3, 103.6e-3, 0.53218)
Br6 = analysis(Br6, 3.4, 0.7, 15.9e6, 0.86e-3, 153.6e-3, 0.53218)
Br7 = analysis(Br7, 3.4, 0.7, 15.9e6, 0.86e-3, 203.2e-3, 0.53218)
Br8 = analysis(Br8, 3.4, 0.7, 15.9e6, 0.86e-3, 250.0e-3, 0.53218)
Br9 = analysis(Br9, 3.4, 0.7, 15.9e6, 0.86e-3, 303.0e-3, 0.53218)

Cu0 = analysis(Cu0, 3.6, 0.7, 58.5e6, 0.88e-3, 0.0, 0.58292)
Cu1 = analysis(Cu1, 3.5, 0.7, 58.5e6, 0.88e-3, 23.7e-3, 0.58292)
Cu2 = analysis(Cu2, 3.5, 0.7, 58.5e6, 0.88e-3, 39.4e-3, 0.58292)
Cu3 = analysis(Cu3, 3.5, 0.7, 58.5e6, 0.88e-3, 66.6e-3, 0.58292)
Cu4 = analysis(Cu4, 3.5, 0.7, 58.5e6, 0.88e-3, 82.8e-3, 0.58292)
Cu5 = analysis(Cu5, 3.5, 0.7, 58.5e6, 0.88e-3, 102.5e-3, 0.58292)
Cu6 = analysis(Cu6, 3.5, 0.7, 58.5e6, 0.88e-3, 136.3e-3, 0.58292)
Cu7 = analysis(Cu7, 3.5, 0.7, 58.5e6, 0.88e-3, 192.8e-3, 0.58292)
Cu8 = analysis(Cu8, 3.5, 0.7, 58.5e6, 0.88e-3, 244.0e-3, 0.58292)
Cu9 = analysis(Cu9, 3.5, 0.7, 58.5e6, 0.88e-3, 297.0e-3, 0.58292)

St0 = analysis(St0, 3.5, 0.7, 3.77e7, 1.07e-3, 0.0, 0.56906 )
St1 = analysis(St1, 3.5, 0.7, 3.77e7, 1.07e-3, 23.7e-3, 0.56906)
St2 = analysis(St2, 3.5, 0.7, 3.77e7, 1.07e-3, 39.4e-3, 0.56906)
St3 = analysis(St3, 3.5, 0.7, 3.77e7, 1.07e-3, 66.6e-3, 0.56906)
St4 = analysis(St4, 3.5, 0.7, 3.77e7, 1.07e-3, 82.8e-3, 0.56906)
St5 = analysis(St5, 3.5, 0.7, 3.77e7, 1.07e-3, 102.5e-3, 0.56906)
St6 = analysis(St6, 3.5, 0.7, 3.77e7, 1.07e-3, 136.3e-3, 0.56906)
St7 = analysis(St7, 3.5, 0.7, 3.77e7, 1.07e-3, 192.8e-3, 0.56906)
St8 = analysis(St8, 3.5, 0.7, 3.77e7, 1.07e-3, 244.0e-3, 0.56906)
St9 = analysis(St9, 3.5, 0.7, 3.77e7, 1.07e-3, 297.0e-3, 0.56906)


# In[91]:


fig1 = plt.figure(1)
fig, ax = plt.subplots()

fig1.set_figheight(8)
fig1.set_figwidth(8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angular Frequency (Rad/s)')

#ax.set_yscale("log")
plt.xlim(0, 10)                    
plt.ylim(120, 180)  

plot1, = plt.plot(St9[t], St9['w'])

handles = [plot0,plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,plot9]
labels = ["B = 0","B = 0.02", "B = 0.04", "B = 0.07", "B = 0.08", "B = 0.10", "B = 0.14", "B = 0.19", "B = 0.24", "B = 0.30"] 

plt.legend(handles,labels, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[ ]:





# In[92]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

display(St9)


# In[111]:


fig1 = plt.figure(1)
fig, ax = plt.subplots()
axis_font = {'fontname':'Arial', 'size':'16'}

fig1.set_figheight(10)
fig1.set_figwidth(8)

plt.yticks(fontsize=16,)
plt.xticks(fontsize=16,)

plt.xlabel('Time (s)',**axis_font)
plt.ylabel('Angular Velocity (Rad/s)',**axis_font)

plt.xlim(0, 40)                    
  
plot0, = plt.plot(Al0[t]-2.0430, Al0['w'])
plot1, = plt.plot(Al1[t]-2.0204, Al1['w'])
plot2, = plt.plot(Al2[t]-1.4318, Al2['w'])
plot3, = plt.plot(Al3[t]-1.4904, Al3['w'])
plot4, = plt.plot(Al4[t]-1.4710, Al4['w'])
plot5, = plt.plot(Al5[t]-2.5056, Al5['w'])
plot6, = plt.plot(Al6[t]-2.1828, Al6['w'])
plot7, = plt.plot(Al7[t]-2.4154, Al7['w'])
plot8, = plt.plot(Al8[t]-2.1192, Al8['w'])
plot9, = plt.plot(Al9[t]-2.8526, Al9['w'])

handles = [plot0,plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,plot9]
labels = ["B = 0.0T","B = 24mT", "B = 39mT", "B = 67mT", "B = 83mT", "B = 103mT", "B = 136mT", "B = 193mT", "B = 244mT", "B = 297mT"] 
plt.legend(handles,labels, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 12})

plt.show()


# In[112]:


fig1 = plt.figure(1)
fig, ax = plt.subplots()
axis_font = {'fontname':'Arial', 'size':'16'}

fig1.set_figheight(10)
fig1.set_figwidth(8)

plt.yticks(fontsize=16,)
plt.xticks(fontsize=16,)

plt.xlabel('Time (s)',**axis_font)
plt.ylabel('Angular Velocity (Rad/s)',**axis_font)

plt.xlim(0, 110)                    
                     
  
plot0, = plt.plot(Br0[t], Br0['w'])
plot1, = plt.plot(Br1[t], Br1['w'])
plot2, = plt.plot(Br2[t], Br2['w'])
plot3, = plt.plot(Br3[t], Br3['w'])
plot4, = plt.plot(Br4[t]-0.2012, Br4['w'])
plot5, = plt.plot(Br5[t], Br5['w'])
plot6, = plt.plot(Br6[t], Br6['w'])
plot7, = plt.plot(Br7[t]-0.4256, Br7['w'])
plot8, = plt.plot(Br8[t], Br8['w'])
plot9, = plt.plot(Br9[t]-0.8606, Br9['w'])

handles = [plot0,plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,plot9]
labels = ["B = 0.0T","B = 40mT", "B = 49mT", "B = 58mT", "B = 81mT", "B = 104mT", "B = 154mT", "B = 203mT", "B = 250mT", "B = 303mT"] 
plt.legend(handles,labels, bbox_to_anchor=(1.05, 1), loc='upper left',  prop={'size': 12})

plt.show()


# In[113]:


fig1 = plt.figure(1)
fig, ax = plt.subplots()
axis_font = {'fontname':'Arial', 'size':'16'}

fig1.set_figheight(10)
fig1.set_figwidth(8)

plt.yticks(fontsize=16,)
plt.xticks(fontsize=16,)

plt.xlabel('Time (s)',**axis_font)
plt.ylabel('Angular Velocity (Rad/s)',**axis_font)

plt.xlim(0, 125)                    
                                   
  
plot0, = plt.plot(Cu0[t]-0.8514, Cu0['w'])
plot1, = plt.plot(Cu1[t]-0.8636, Cu1['w'])
plot2, = plt.plot(Cu2[t]-1.1078, Cu2['w'])
plot3, = plt.plot(Cu3[t]-2.0154, Cu3['w'])
plot4, = plt.plot(Cu4[t]-1.9244, Cu4['w'])
plot5, = plt.plot(Cu5[t]-1.5460, Cu5['w'])
plot6, = plt.plot(Cu6[t]-0.2564, Cu6['w'])
plot7, = plt.plot(Cu7[t]-0.2520, Cu7['w'])
plot8, = plt.plot(Cu8[t]-1.3754, Cu8['w'])
plot9, = plt.plot(Cu9[t]-1.0228, Cu9['w'])

handles = [plot0,plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,plot9]
labels = ["B = 0.0T","B = 38mT", "B = 62mT", "B = 82mT", "B = 106mT", "B = 150mT", "B = 204mT", "B = 254mT", "B = 288mT", "B = 302mT"] 
plt.legend(handles,labels, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 12})

plt.show()


# In[115]:


fig1 = plt.figure(1)
fig, ax = plt.subplots()
axis_font = {'fontname':'Arial', 'size':'16'}

fig1.set_figheight(10)
fig1.set_figwidth(8)

plt.yticks(fontsize=16,)
plt.xticks(fontsize=16,)

plt.xlabel('Time (s)',**axis_font)
plt.ylabel('Angular Velocity (Rad/s)',**axis_font)

plt.xlim(0, 120)                    
                                     
  
plot0, = plt.plot(St0[t], St0['w'])
plot1, = plt.plot(St1[t]-0.7128, St1['w'])
plot2, = plt.plot(St2[t]-0.5760, St2['w'])
plot3, = plt.plot(St3[t]-2.0234, St3['w'])
plot4, = plt.plot(St4[t]-0.5796, St4['w'])
plot5, = plt.plot(St5[t]-1.1782, St5['w'])
plot6, = plt.plot(St6[t]-1.2082, St6['w'])
plot7, = plt.plot(St7[t], St7['w'])
plot8, = plt.plot(St8[t]-0.2548, St8['w'])
plot9, = plt.plot(St9[t]-1.2158, St9['w'])

handles = [plot0,plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,plot9]
labels = ["B = 0.0T","B = 38mT", "B = 50mT", "B = 60mT", "B = 83mT", "B = 105mT", "B = 149mT", "B = 203mT", "B = 247mT", "B = 302mT"] 
plt.legend(handles,labels, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 12})

plt.show()


# In[118]:


fig1 = plt.figure(1)
fig, ax = plt.subplots()
axis_font = {'fontname':'Arial', 'size':'16'}

fig1.set_figheight(10)
fig1.set_figwidth(8)

plt.yticks(fontsize=16,)
plt.xticks(fontsize=16,)

plt.xlabel('Time (s)',**axis_font)
plt.ylabel('Angular Velocity (Rad/s)',**axis_font)

plt.xlim(0, 125)                    
                                     
  
plot0, = plt.plot(Al0[t]-2.0430, Al0['w'])
plot1, = plt.plot(Br0[t], Br0['w'])
plot2, = plt.plot(Cu0[t]-0.8514, Cu0['w'])
plot3, = plt.plot(St0[t], St0['w'])


handles = [plot0,plot1,plot2,plot3]
labels = ["Aluminium", "Brass", "Copper", "Steel"] 
plt.legend(handles,labels, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 12})

plt.show()


# In[120]:


fig1 = plt.figure(1)
fig, ax = plt.subplots()
axis_font = {'fontname':'Arial', 'size':'16'}

fig1.set_figheight(10)
fig1.set_figwidth(8)

plt.yticks(fontsize=16,)
plt.xticks(fontsize=16,)

plt.xlabel('Time (s)',**axis_font)
plt.ylabel('Angular Velocity (Rad/s)',**axis_font)

plt.xlim(0, 40)                    
                                     
  
plot0, = plt.plot(Al9[t]-2.8526, Al9['w'])
plot1, = plt.plot(Br9[t]-0.8606, Br9['w'])
plot2, = plt.plot(Cu9[t]-1.0228, Cu9['w'])
plot3, = plt.plot(St9[t]-1.2158, St9['w'])


handles = [plot0,plot1,plot2,plot3]
labels = ["Aluminium", "Brass", "Copper", "Steel"] 
plt.legend(handles,labels, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 12})

plt.show()


# In[158]:


fig1 = plt.figure(1)
fig, ax = plt.subplots()

fig1.set_figheight(8)
fig1.set_figwidth(8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angular Frequency (Rad/s)')

plt.xlim(0, 10)  
plt.ylim(0, 200)

w = Cu5['w']
t1 = Cu5[t]

plt.plot(t1, w)

def func(t1,a,b):
    return a*np.exp(-b*t1**3)
popt, pcov = curve_fit(func, t1, w)
line = np.linspace(-10,20,200)
plt.plot(line, func(line, *popt))

plt.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





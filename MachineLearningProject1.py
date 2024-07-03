import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
import random

#Load data into X,y format.
filename = 'C:/Users/jensl/Desktop/Python Files/spambase/spambase.data'
n_attributes = 57
df = pd.read_csv(filename)
raw_data = df.values
cols = range(0, n_attributes)
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])
y = raw_data[:,-1]
N, M = X.shape
C = len(np.unique(y))
classNames = ['Not spam', 'Spam']

#Summary statistics.

#Print latex table with summary statistics.
print("\\begin{longtable}{|l|l|l|l|l|l|}")
print("\hline")
print("& \\textbf{Mean} & \\textbf{Std. dev.} & \\textbf{Median} & \\textbf{Min} & \\textbf{Max} \\\ \hline")
print("\endfirsthead")
print("%")
print("\endhead")
print("%")
for i in range(0, n_attributes):
    print("\\textbf{"+(attributeNames[i]).replace("_", "\_")+"} & "+str(round(X[:,i].mean(),5))+" & "+str(round(X[:,i].std(ddof=1),5))+" & "+str(np.median(X[:,i]))+
          " & "+str(X[:,i].min())+" & "+str(X[:,i].max())+" \\\ \hline")
print("\end{longtable}")

print(list(y).count(0))
print(list(y).count(1))

#Plot histograms.
hist_indices = [0, 10, 20, 26, 48, 49, 50, 51, 54, 55, 56]
plt.figure(figsize=(14,9))
u = np.floor(np.sqrt(len(hist_indices))); v = np.ceil(float(len(hist_indices))/u)
for i in range(len(hist_indices)):
    plt.subplot(int(u),int(v),i+1)
    plt.hist(X[:,hist_indices[i]],bins=20,range=(X[:,hist_indices[i]].min(),X[:,hist_indices[i]].min()+2*X[:,hist_indices[i]].std()))
    plt.xlabel(attributeNames[hist_indices[i]])
    plt.ylim(0, N)
    if i%v!=0: plt.yticks([])
    if i==0: plt.title('Spam: Histogram')
plt.tight_layout()
plt.show()

#Correlation.
plt.matshow(np.corrcoef(X.T))
plt.colorbar()
plt.title('Spam: Correlation matrix')
plt.show()

#Subtract mean
Y = X - np.ones((N, 1))*X.mean(0)

Y = Y*(1/np.std(Y,0)) #Standardization

#Box plots
plt.boxplot(Y)
ticks = np.empty(57, dtype=object)
ticks[:] = ''
for i in range(4, 57, 5):
    ticks[i] = str(i+1)
plt.xticks(range(1,58),ticks)
plt.xlabel('Attribute')
plt.ylabel('Normalized value')
plt.title('Spam: Boxplots')
plt.show()

#Perform PCA
U,S,Vh = svd(Y,full_matrices=False)
V = Vh.T
#Explained variance
rho = (S*S) / (S*S).sum() 

#Plot variance explained
threshold = 0.9
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#Projection of data
Z = Y @ V
#Plot of PCA
#Indices of the principal components to be plotted
i = 0
j = 2
f = plt.figure()
plt.title('Spam data: PCA')
#indices = random.sample(range(0, N), 100)
#Z2 = Z[indices,:]
#y2 = y[indices]
for c in range(C):
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.show()

#Plot PC coefficients
N,M = X.shape
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
#plt.xticks(r+bw, attributeNames)
plt.xlabel('Attribute')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Spam data: PCA Component Coefficients')
plt.show()
#print(V[:,0].T) #Print PC2
#print(attributeNames[V[:,0]>0.20])
#print(V[V[:,0]>0.20,0])
#print(attributeNames[V[:,1]>0.20])
print(attributeNames[V[:,2]<-0.10])
print(attributeNames[V[:,2]>0.20])

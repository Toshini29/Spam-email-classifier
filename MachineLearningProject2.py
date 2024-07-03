import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import model_selection
import torch
from toolbox_02450 import train_neural_net, correlated_ttest
from sklearn.linear_model import LogisticRegression, Ridge

#Load data into X,y format.
filename = 'C:/Users/jensl/Desktop/Python Files/spambase/spambase.data'
n_attributes = 57
df = pd.read_csv(filename)
raw_data = df.values


#####Regression part a#####
cols = range(0, n_attributes-3) #Don't include the last three attributes.
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])
y = raw_data[:,-2] #capital_run_length_total
N, M = X.shape

# Normalize data
X = stats.zscore(X)

lambdas = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]
K = 10
errors_lr = np.zeros((K, len(lambdas)))
errors_base = np.zeros(K)
CV = model_selection.KFold(K, shuffle=True)
for i, (train_index, test_index) in enumerate(CV.split(X, y)):
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    j = 0
    for l in range(0, len(lambdas)):
        l = lambdas[j]
        model = Ridge(alpha=l)
        model = model.fit(X_train,y_train)
        y_est = model.predict(X_test)
        test_error = np.power(y_test-y_est,2).mean()
        errors_lr[i,j] = test_error*len(y_test)/X.shape[0]
        j += 1
    y_mean = np.mean(y_train)
    base = np.zeros(len(y_test))
    base[:] = y_mean
    test_error = np.power(y_test-base,2).mean()
    errors_base[i] = test_error*len(y_test)/X.shape[0]
    print(f"Fold {i+1} of {K} done")
gen_errors_lr = np.sum(errors_lr,0)
gen_error_base = np.sum(errors_base)
print(f"Lambdas: {lambdas}")
print(f"Gen errors: {gen_errors_lr}")
print(f"Base gen error: {gen_error_base}")

plt.plot(lambdas, gen_errors_lr)
plt.xlabel("Lambda")
plt.ylabel("Generalization error")
plt.title("Generalization error as function of lambda")
plt.show()

best_index = np.argmin(gen_errors_lr)
best_lambda = lambdas[best_index]
best_gen_error = gen_errors_lr[best_index]
print(f"Best lambda: {best_lambda}")
print(f"Best gen error: {best_gen_error}")

model = Ridge(alpha=best_lambda)
model = model.fit(X,y)
print(f"Weights: {model.coef_}")
print(f"Intercept: {model.intercept_}")

threshold = np.sort(np.abs(model.coef_))[-17]
mask = np.abs(model.coef_) >= threshold
coef = model.coef_[mask]
print(len(coef))
plt.bar([i for i in range(0,M)],model.coef_)
plt.title("Spam base: All coefficients for LR model")
plt.xlabel("Attribute")
plt.ylabel("Coefficient")
plt.show()

plt.bar(attributeNames[mask],coef)
plt.title("Spam base: Important coefficients for LR model")
plt.xlabel("Attribute")
plt.ylabel("Coefficient")
plt.xticks(rotation=90)
plt.show()


#####Regression part b#####

#Parameters for neural network classifier
n_replicates = 1
max_iter = 10000

# K-fold crossvalidation for ANN, Linear Regression and baseline
K = 10
h_range = [1, 300, 400, 500]
lambda_interval = lambdas
loss_fn = torch.nn.MSELoss()

CV = model_selection.KFold(K, shuffle=True)
result_errors_ann = np.zeros(K)
result_h = np.zeros(K)
result_errors_lr = np.zeros(K)
result_lambda = np.zeros(K)
result_errors_base = np.zeros(K)
rAL = []
rAB = []
rLB = []

for k, (train_index, test_index) in enumerate(CV.split(X,y)):
    print('Outer fold: {0}/{1}'.format(k+1,K))    
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    Ki = 10
    CVi = model_selection.KFold(Ki, shuffle=True)
    errors_ann = np.ones((Ki,len(h_range)))
    errors_lr = np.ones((Ki,len(lambda_interval)))
    
    for i, (inner_train_index, inner_test_index) in enumerate(CVi.split(X_train, y_train)):
        X_train_inner = X_train[inner_train_index,:]
        y_train_inner = y_train[inner_train_index]
        X_test_inner = X_train[inner_test_index,:]
        y_test_inner = y_train[inner_test_index]
        
        j = 0
        for h in h_range:
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, h), #M features to h
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h, 1), # h to 1 output neuron
                                )
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=torch.Tensor(X_train_inner),
                                                               y=torch.Tensor(y_train_inner).unsqueeze(1),
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            y_test_inner_tensor = torch.Tensor(y_test_inner).unsqueeze(1)
            y_test_est = net(torch.Tensor(X_test_inner))
            se = (y_test_est.data.numpy().squeeze()-y_test_inner)**2 # squared error
            mse = sum(se)/len(y_test_inner) #mean
            errors_ann[i,j] = mse*len(y_test_inner)/X_train.shape[0]
            print(f"Error rate: {mse}")
            j += 1
        for l in range(0, len(lambda_interval)):
            model = Ridge(alpha=lambda_interval[l])
            model = model.fit(X_train_inner,y_train_inner)
            y_test_est = model.predict(X_test_inner)
            se = (y_test_est-y_test_inner)**2 # squared error
            mse = sum(se)/len(y_test_inner) #mean
            errors_lr[i,l] = mse*len(y_test_inner)/X_train.shape[0]
            print(f"Error rate: {mse}")
    gen_errors_ann = np.sum(errors_ann,0)
    max_index = np.argmin(gen_errors_ann)
    result_h[k] = h_range[max_index]
    print(f"Inner fold {k+1} best h: {h_range[max_index]}")
    print(f"Inner fold {k+1} best error ann: {gen_errors_ann[max_index]}")
    gen_errors_lr = np.sum(errors_lr,0)
    max_index = np.argmin(gen_errors_lr)
    result_lambda[k] = lambda_interval[max_index]
    print(f"Inner fold {k+1} best lambda: {lambda_interval[max_index]}")
    print(f"Inner fold {k+1} best error lr: {gen_errors_lr[max_index]}")
    
    #For statistical evaluation
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, h), #M features to h
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(h, 1), # h to 1 output neuron
                        )
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=torch.Tensor(X_train),
                                                       y=torch.Tensor(y_train).unsqueeze(1),
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    model = Ridge(alpha=result_lambda[k])
    model = model.fit(X_train,y_train)
    yhatL = model.predict(X_test).T
    se = (yhatL-y_test)**2 # squared error
    result_errors_lr[k] = sum(se)/len(y_test) #mean
    
    y_test_est = net(torch.Tensor(X_test))
    se = (y_test_est.data.numpy().squeeze()-y_test)**2 # squared error
    result_errors_ann[k] = sum(se)/len(y_test) #mean
    
    y_mean = np.mean(y_train)
    base = np.zeros(len(y_test))
    base[:] = y_mean
    result_errors_base[k] = np.power(y_test-base,2).mean()
    
    yhatA = y_test_est.type(dtype=torch.float64).data.numpy().squeeze()
    yhatB = base
    rAB.append( np.mean( np.abs( yhatA-y_test ) - np.abs( yhatB-y_test) ) )
    rAL.append( np.mean( np.abs( yhatA-y_test ) - np.abs( yhatL-y_test) ) )
    rLB.append( np.mean( np.abs( yhatL-y_test ) - np.abs( yhatB-y_test) ) )
print(f"Result errors ann: {result_errors_ann}")
print(f"Result h: {result_h}")
print(f"Result errors lr: {result_errors_lr}")
print(f"Result lambda: {result_lambda}")
print(f"Result errors base: {result_errors_base}")

#Statistical evaluation
alpha = 0.05
rho = 1/K
p_setupII_AB, CI_setupII_AB = correlated_ttest(rAB, rho, alpha=alpha)
p_setupII_AL, CI_setupII_AL = correlated_ttest(rAL, rho, alpha=alpha)
p_setupII_LB, CI_setupII_LB = correlated_ttest(rLB, rho, alpha=alpha)
print(f"CI for ANN and Base: {CI_setupII_AB}")
print(f"p-value for ANN and Base: {p_setupII_AB}")
print(f"CI for ANN and Linear Regression: {CI_setupII_AL}")
print(f"p-value for ANN and Linear Regression: {p_setupII_AL}")
print(f"CI for Linear Regression and Base: {CI_setupII_LB}")
print(f"p-value for Linear Regression and Base: {p_setupII_LB}")




#####Classification#####
cols = range(0, n_attributes)
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])
y = raw_data[:,-1]
N, M = X.shape
C = len(np.unique(y))
classNames = ['Not spam', 'Spam']

# Normalize data
X = stats.zscore(X)

# Parameters for neural network classifier
n_replicates = 1        # number of networks trained
max_iter = 10000

h_range = [1, 2, 3, 10]
lambda_interval = np.power(10.,range(-5, 5))
loss_fn = torch.nn.BCELoss()

# K-fold crossvalidation for ANN, Logistic Regression and baseline
K = 10
CV = model_selection.KFold(K, shuffle=True)
result_errors_ann = np.zeros(K)
result_h = np.zeros(K)
result_errors_lr = np.zeros(K)
result_lambda = np.zeros(K)
result_errors_base = np.zeros(K)
rAL = []
rAB = []
rLB = []

for k, (train_index, test_index) in enumerate(CV.split(X,y)):
    print('Outer fold: {0}/{1}'.format(k+1,K))    
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    Ki = 10
    CVi = model_selection.KFold(Ki, shuffle=True)
    errors_ann = np.ones((Ki,len(h_range)))
    errors_lr = np.ones((Ki,len(lambda_interval)))
    
    for i, (inner_train_index, inner_test_index) in enumerate(CVi.split(X_train, y_train)):
        X_train_inner = X_train[inner_train_index,:]
        y_train_inner = y_train[inner_train_index]
        X_test_inner = X_train[inner_test_index,:]
        y_test_inner = y_train[inner_test_index]
        
        j = 0
        for h in h_range:
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, h), #M features to h
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h, 1), # h to 1 output neuron
                                torch.nn.Sigmoid()
                                )
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=torch.Tensor(X_train_inner),
                                                               y=torch.Tensor(y_train_inner).unsqueeze(1),
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            y_test_inner_tensor = torch.Tensor(y_test_inner).unsqueeze(1)
            y_sigmoid = net(torch.Tensor(X_test_inner))
            y_test_est = (y_sigmoid>.5).type(dtype=torch.uint8)
            y_test_inner_t = y_test_inner_tensor.type(dtype=torch.uint8)
            e = y_test_est != y_test_inner_t
            error_rate = (sum(e).type(torch.float)/len(y_test_inner_t)).data.numpy()[0]
            errors_ann[i,j] = error_rate*len(y_test_inner)/X_train.shape[0]
            e_sum = sum(e).type(torch.float).data.numpy()[0]
            print(f"Error: {e_sum}/{len(y_test_inner)}")
            print(f"Error rate: {error_rate}")
            j += 1
        for l in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[l], max_iter=1000)
            mdl.fit(X_train_inner, y_train_inner)
            y_test_est = mdl.predict(X_test_inner).T
            error_rate = np.sum(y_test_est != y_test_inner) / len(y_test_inner)
            errors_lr[i,l] = error_rate*len(y_test_inner)/X_train.shape[0]
            e_sum = np.sum(y_test_est != y_test_inner)
            print(f"Error: {e_sum}/{len(y_test_inner)}")
            print(f"Error rate: {error_rate}")
    gen_errors_ann = np.sum(errors_ann,0)
    max_index = np.argmin(gen_errors_ann)
    result_h[k] = h_range[max_index]
    print(f"Inner fold {k+1} best h: {h_range[max_index]}")
    print(f"Inner fold {k+1} best error ann: {gen_errors_ann[max_index]}")
    gen_errors_lr = np.sum(errors_lr,0)
    max_index = np.argmin(gen_errors_lr)
    result_lambda[k] = lambda_interval[max_index]
    print(f"Inner fold {k+1} best lambda: {lambda_interval[max_index]}")
    print(f"Inner fold {k+1} best error lr: {gen_errors_lr[max_index]}")
    
    #For statistical evaluation
    model_ann = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, int(result_h[k])), #M features to h
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(int(result_h[k]), 1), # h to 1 output neuron
                        torch.nn.Sigmoid()
                        )
    net, final_loss, learning_curve = train_neural_net(model_ann,
                                                       loss_fn,
                                                       X=torch.Tensor(X_train),
                                                       y=torch.Tensor(y_train).unsqueeze(1),
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    mdl = LogisticRegression(penalty='l2', C=1/result_lambda[k], max_iter=1000)
    mdl.fit(X_train, y_train)
    yhatL = mdl.predict(X_test).T
    result_errors_lr[k] = np.sum(yhatL != y_test) / len(y_test)
    
    y_sigmoid = net(torch.Tensor(X_test))
    y_test_est = (y_sigmoid>.5).type(dtype=torch.uint8)
    y_test_t = torch.Tensor(y_test).unsqueeze(1).type(dtype=torch.uint8)
    e = y_test_est != y_test_t
    result_errors_ann[k] = (sum(e).type(torch.float)/len(y_test_t)).data.numpy()[0]
    
    most_common_class = 0
    if sum(y_train) > len(y_train)-sum(y_train):
        most_common_class = 1
    base = np.zeros(len(y_test))
    base[:] = most_common_class
    result_errors_base[k] = np.sum(base != y_test) / len(y_test)
    
    yhatA = (y_sigmoid>.5).type(dtype=torch.float64).data.numpy().squeeze()
    yhatB = base
    rAB.append( np.mean( np.abs( yhatA-y_test ) - np.abs( yhatB-y_test) ) )
    rAL.append( np.mean( np.abs( yhatA-y_test ) - np.abs( yhatL-y_test) ) )
    rLB.append( np.mean( np.abs( yhatL-y_test ) - np.abs( yhatB-y_test) ) )
print(f"Result errors ann: {result_errors_ann}")
print(f"Result h: {result_h}")
print(f"Result errors lr: {result_errors_lr}")
print(f"Result lambda: {result_lambda}")
print(f"Result errors base: {result_errors_base}")

#Statistical evaluation
alpha = 0.05
rho = 1/K
p_setupII_AB, CI_setupII_AB = correlated_ttest(rAB, rho, alpha=alpha)
p_setupII_AL, CI_setupII_AL = correlated_ttest(rAL, rho, alpha=alpha)
p_setupII_LB, CI_setupII_LB = correlated_ttest(rLB, rho, alpha=alpha)
print(f"CI for ANN and Base: {CI_setupII_AB}")
print(f"p-value for ANN and Base: {p_setupII_AB}")
print(f"CI for ANN and Logistic Regression: {CI_setupII_AL}")
print(f"p-value for ANN and Logistic Regression: {p_setupII_AL}")
print(f"CI for Logistic Regression and Base: {CI_setupII_LB}")
print(f"p-value for Logistic Regression and Base: {p_setupII_LB}")



#Logistic regression
l = 0.1
mdl = LogisticRegression(penalty='l2', C=1/l, max_iter=1000)
mdl.fit(X, y)
print(f"Coefficients: {mdl.coef_}")
print(f"Intercept: {mdl.intercept_}")
mask = np.abs(mdl.coef_[0]) > 0.8
coef = mdl.coef_[0,mask]
plt.bar([i for i in range(0,M)],mdl.coef_[0])
plt.title("Spam base: All coefficients for LR model")
plt.xlabel("Attribute")
plt.ylabel("Coefficient")
plt.show()

plt.bar(attributeNames[mask],coef)
plt.title("Spam base: Important coefficients for LR model")
plt.xlabel("Attribute")
plt.ylabel("Coefficient")
plt.xticks(rotation=90)
plt.show()

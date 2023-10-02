'''Logistic regression test and display.'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Machine learning
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import accuracy_score

#def sigmoid(theta):
#    '''Sigmoid function.'''

#    sigmoid = 1 / (1 + np.exp(-theta))
#    return sigmoid


#def loss(h, y):
#    '''Loss function. '''

#    loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
#    return loss


#def gradient(x, h, y):
#    '''Gradient calculation. '''

#    gradient = np.dot(x.T, (h - y)) / y.shape[0]
#    return gradient


#def Logistic_Regression(x, y, lr, num_iter):
#    '''Logistic regression process. '''

#    intercept = np.ones((x.shape[0], 1))
#    x = np.concatenate((intercept, x), axis = 1)
#    w = np.zeros(x.shape[1])

#    # Store loss value.
#    l_list = []
#    for i in range(num_iter):
#        z = np.dot(x, w)
#        h = sigmoid(z)

#        g = gradient(x, h, y)
#        w -= lr * g

#        z = np.dot(x, w)
#        h = sigmoid(z)

#        l = loss(h, y)
#        l_list.append(l)

#    return l, w, l_list


#"""Main function."""
#plt.rcParams['axes.unicode_minus'] = False
#df = pd.read_csv("logisticdata.csv", header = 0)
#df.head()
#print(df)

#x = df[['PM2.5','No2']].values
#y = df['Flag'].values

## Fitting
#l, w, l_y = Logistic_Regression(x, y, lr = 0.0005, num_iter = 5000)
#L = l, w
#print(w)

## Draw the results
#plt.scatter(df['PM2.5'], df['No2'], c = df['Flag'], label = 'Samples')

#x1_min, x1_max = df['PM2.5'].max(), df['PM2.5'].min(),
#x2_min, x2_max = df['No2'].min(), df['No2'].max(),

#xx1, xx2 = np.meshgrid(np.linspace(x1_max, x1_min), np.linspace(x2_max, x2_min))
#grid = np.c_[xx1.ravel(), xx2.ravel()]

#probs = (np.dot(grid, np.array([L[1][1:5]]).T) + L[1][0]).reshape(xx1.shape)
#plt.contour(xx1, xx2, probs, levels = [0], linewidths = 1, colors = 'red');
#plt.xlabel('PM2.5')
#plt.ylabel('No2')
#plt.title('Classification of pollution level in terms of PM2.5 and No2')
#plt.show()

#plt.plot([i for i in range(len(l_y))], l_y)
#plt.title('PM2.5 and No2 logistic regression')
#plt.xlabel("Interative times")
#plt.ylabel("Loss function")
#plt.show()

#def loadData():
#	train_x = []
#	train_y = []
#	fileIn = open('logisticdata_func.txt')
#	for line in fileIn.readlines():
#		lineArr = line.strip().split()
#		train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
#		train_y.append(float(lineArr[2]))
#	return np.mat(train_x), np.mat(train_y).transpose()

##[-0.26745288  0.07470762 -0.11586046]
#def showLogRegres(weights, train_x, train_y):
#	# notice: train_x and train_y is mat datatype
#	numSamples, numFeatures = np.shape(train_x)
#	if numFeatures != 3:
#		print("Sorry! I can not draw because the dimension of your data is not 2!")
#		return 1

#	# draw all samples
#	for i in range(numSamples):
#		if int(train_y[i, 0]) == 0:
#			plt.plot(train_x[i, 1], train_x[i, 2], 'or')
#		elif int(train_y[i, 0]) == 1:
#			plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

#	# draw the classify line
#	min_x = min(train_x[:, 1])[0, 0]
#	max_x = max(train_x[:, 1])[0, 0]
#	#weights = weights.getA()  # convert mat to array
#	y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
#	y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
#	plt.plot([min_x, max_x], [y_min_x, y_max_x], color ='#F4A460')
#	plt.xlabel('X1'); plt.ylabel('X2')
#	plt.show()


def test_logistic():
    df = pd.read_csv("logisticdata2021.csv", header = 0)

    X = df[['PM2.5','No2']].values
    y = df['Flag'].values

    #X, y = loadData()

    # Partition
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 1)

    #plt.scatter(X_train[[0]] ,X_train[[1]] , color = '#7B68EE', alpha = 0.7, label = 'Training samples')
    #plt.plot(X_train , logistic_reg.predict(X_train), color ='#F4A460', label = 'Fitting curve')
    #plt.xlabel('$PM_{2.5} \quad (\mu g/m^{3})$')
    #plt.ylabel('$NO_{2} \quad (\mu g/m^{3})$')
    #plt.title('Training set: $R^{2}=$' + str(logistic_reg.score(X_train, y_train)))
    #plt.legend()
    #plt.show()
    #print('\nTrain Score: ', logistic_reg.score(X_train, y_train))

    # Training set
    l1_train_predict = []
    l2_train_predict = []

    # Testing set
    l1_test_predict = []
    l2_test_predict = []

    # Loop 50 inverese regularization coefficients from 0.01 to 2.
    for regular_coef in np.linspace(0.01, 2, 50) :
        lr_l1 = linear_model.LogisticRegression(penalty = 'l1', C = regular_coef, solver = 'liblinear', multi_class = 'auto', max_iter = 2000)
        lr_l2 = linear_model.LogisticRegression(penalty = 'l2', C = regular_coef, solver = 'liblinear', multi_class = 'auto', max_iter = 2000)

        # L1 regularization, append each ACC in training set and testing set, respectively.
        lr_l1.fit(X_train, y_train)
        l1_train_predict.append(accuracy_score(y_train, lr_l1.predict(X_train)))
        l1_test_predict.append(accuracy_score(y_test, lr_l1.predict(X_test)))

        # L2 regularization, append each ACC in training set and testing set, respectively.
        lr_l2.fit(X_train, y_train)
        l2_train_predict.append(accuracy_score(y_train, lr_l2.predict(X_train)))
        l2_test_predict.append(accuracy_score(y_test, lr_l2.predict(X_test)))

    print('\nL1 Regularization:')
    print('Train Score: ', lr_l1.score(X_train, y_train))
    print('Train Accuracy: ', l1_train_predict)
    print('Test Score: ', lr_l1.score(X_test, y_test))
    print('Test Accuracy: ', l1_test_predict)
    print('\nL2 Regularization:')
    print('Train Score: ', lr_l2.score(X_train, y_train))
    print('Test Score: ', lr_l2.score(X_test, y_test))     
    # print(l2_test_predict)

    plt.style.use(['science', 'no-latex'])
    plt.figure(dpi = 165)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['mathtext.default'] = 'regular'

    data = [l1_train_predict, l2_train_predict, l1_test_predict, l2_test_predict]
    label = ['L1_train', 'L2_train', 'L1_test', "L2_test"]
    color = ['#2A557F', '#45BC9C', '#F05076', '#FFCD6E']

    for i in range(4) :
        plt.plot(np.linspace(0.01, 2, 50), data[i], label = label[i],
                color = color[i], linewidth = 1.3, marker = '.', markersize = 3.5)

    # The smaller, the stronger the regularization is.
    plt.xlabel('Inverse of regularization strength')
    # The percentage of correctly classified samples.
    plt.ylabel('Accuracy')
    plt.grid(visible = True, which = 'major', linestyle = '-', color = '#C0C0C0')
    plt.grid(visible = True, which = 'minor', linestyle = '--', color = '#C0C0C0', alpha = 0.5)
    plt.legend(loc = 'lower right')
    plt.title('(d)')
    plt.show()


def logistic_classification():
    data = pd.read_csv('logisticdata.csv')

    X = np.array(data[['PM2.5','No2']])
    y = np.array(data['Flag'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 1)
    # logistic_reg = linear_model.LogisticRegression(penalty = 'l2', C = 1.8, solver = 'liblinear', multi_class = 'auto', max_iter = 2000)
    logistic_reg = linear_model.LogisticRegressionCV(penalty = 'l2', solver = 'liblinear', multi_class = 'auto', max_iter = 2000)
    logistic_reg.fit(X_train, y_train)

    print('w1, w2: ', logistic_reg.coef_) # w1, w2....wn
    print('w0: ', logistic_reg.intercept_) # w0
    print('Train Accuracy: ', accuracy_score(y_train, logistic_reg.predict(X_train)))
    print('Test Accuracy: ', accuracy_score(y_test, logistic_reg.predict(X_test)))

    #X1 = data[data['Flag'] == 0]
    #X2 = data[data['Flag'] == 1]
    X1 = X_train[y_train == 0]
    X2 = X_train[y_train == 1]
    #print('y_train: ', y_train)
    #print('X1: ', X1)
    #print('X1_1: ', X1[:, 0])
    #print('X2: ', X2)

    plt.style.use(['science', 'no-latex'])
    #plt.figure(figsize = (10, 8), dpi = 210)
    plt.figure(dpi = 165)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['mathtext.default'] = 'regular'

    plt.subplot(1, 2, 1)
    #plt.scatter(X2['PM2.5'], X2['No2'], color = 'red', label = 'Polluted')
    #plt.scatter(X1['PM2.5'], X1['No2'], color = 'blue', label = 'Un-polluted')
    plt.scatter(X2[:, 0], X2[:, 1], color = '#FEA3A2', label = 'Polluted samples', alpha = 0.7)
    plt.scatter(X1[:, 0], X1[:, 1], color = '#45BC9C', label = 'Unpolluted samples', alpha = 0.7)

    x1_plot = np.linspace(50, 84)
    #print('X1_plot: ', x1_plot)
    x2_plot = (-logistic_reg.coef_[0][0] * x1_plot - logistic_reg.intercept_[0]) / logistic_reg.coef_[0][1]
    #print('X2_plot: ', x2_plot)
    plt.plot(x1_plot, x2_plot, color = '#8E8BFE',
            label = 'Decision boundary:\n $h_{w}(x)=0.1202x_{1}-0.03169x_{2}-6.548$')
    '''
        $h_{w}(x)=0.1543x_{1}-0.04723x_{2}-7.829$
        $h_{w}(x)=0.1153x_{1}-0.04326x_{2}-5.761$
        $h_{w}(x)=0.1417x_{1}-0.09610x_{2}-5.952$
        $h_{w}(x)=0.1202x_{1}-0.03169x_{2}-6.548$
    '''
    # plt.title('Training set: $R^{2}=$' + str('{:.4f}'.format(logistic_reg.score(X_train, y_train))))
    plt.title('(a)')
    plt.ylim((-5, 150))
    plt.xlabel('$PM_{2.5}$ $(\mu g/m^{3})$')
    plt.ylabel('$NO_{2}$ $(\mu g/m^{3})$')
    plt.legend(loc = 'upper left')
    #plt.show()

    X1 = X_test[y_test == 0]
    X2 = X_test[y_test == 1]

    plt.subplot(1, 2, 2)
    plt.scatter(X2[:, 0], X2[:, 1], color = '#FEA3A2', label = 'Polluted samples', alpha = 0.7)
    plt.scatter(X1[:, 0], X1[:, 1], color = '#45BC9C', label = 'Unpolluted samples', alpha = 0.7)

    x1_plot = np.linspace(50, 83)
    x2_plot = (-logistic_reg.coef_[0][0] * x1_plot - logistic_reg.intercept_[0]) / logistic_reg.coef_[0][1]
    plt.plot(x1_plot, x2_plot, color = '#8E8BFE',
            label = 'Decision boundary:\n $h_{w}(x)=0.1202x_{1}-0.03169x_{2}-6.548$')
    # plt.title('Test set: $R^{2}=$' + str('{:.4f}'.format(logistic_reg.score(X_test, y_test))))
    plt.title('(b)')
    plt.ylim((-5, 150))
    plt.xlabel('$PM_{2.5}$ $(\mu g/m^{3})$')
    plt.ylabel('$NO_{2}$ $(\mu g/m^{3})$')
    plt.legend(loc = 'upper left')
    plt.show()


#print(loadData())
#test_logistic()
logistic_classification()

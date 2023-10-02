'''Linear regression test and display.'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Linear regression and cross-validation in machine learning.
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

# Polynomial regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

def linear_regression():
    '''Linear regression.'''
    print('Linear regression:\n')

    # Read data and print it.
    datalist = pd.read_csv('lineardata2021.csv')
    X = datalist.iloc[ : , : 1].values
    Y = datalist.iloc[ : , 1].values
    print(datalist)

    # Convert to array format.
    X = np.array(datalist[['PM2.5']])
    Y = np.array(datalist['No2'])
   
    plt.style.use(['science', 'no-latex'])
    plt.figure(dpi = 165)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['mathtext.default'] = 'regular'

    # Draw a scatter graph, with blue circular markers.
    #plt.scatter(X, Y, 29, color = 'blue', marker = 'o', linewidth = 1.9, alpha = 0.7)
    #plt.xlabel('PM2.5')
    #plt.ylabel('No2')
    #plt.title('The PM2.5 and No2 in Beijing in 2017-2021')
    #plt.grid(color = '#5F9EA0', linestyle = '-.', linewidth = 1.2, axis = 'both', alpha = 0.3)
    #plt.show()

    # 80% training samples while 20% testing samples.
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 1)

    # Introduce the training set to the linear regression model (Least Square Method).
    linear_reg = linear_model.LinearRegression()
    linear_reg.fit(x_train, y_train)
    
    # Display slope and intercept of the linear regression model.
    print('\nPrint slope and intercept of the linear regression model: ')
    print('Slope: ', linear_reg.coef_)
    print('Intercept: ', linear_reg.intercept_)
    # Return the coefficient of determination R^2 of the prediction.
    print('\nTrain Score: ', linear_reg.score(x_train, y_train))
    print('Test Score: ', linear_reg.score(x_test, y_test))

    plt.subplot(1, 2, 1)
    # Visualization of training results.
    plt.scatter(x_train , y_train, color = '#45BC9C', alpha = 0.7, label = 'Testing samples')
    plt.plot(x_train , linear_reg.predict(x_train), color = '#2A557F',
            label = 'Fitting curve: $y=0.2483x+16.97$')
    '''
        $y=0.3196x+25.64$
        $y=0.3626x+20.89$
        $y=0.2483x+16.97$
        $y=0.2915x+21.42$
    '''
    #x1_plot = np.linspace(0, 350)
    #x2_plot = linear_reg.coef_ * x1_plot + linear_reg.intercept_
    #plt.plot(x1_plot, x2_plot, color = 'r')
    plt.xlabel('$PM_{2.5}$ $(\mu g/m^{3})$')
    plt.ylabel('$NO_{2}$ $(\mu g/m^{3})$')
    plt.title('(a)')
    # plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0)
    # plt.subplots_adjust(right = 0.7)
    plt.legend(loc = 'upper left')

    plt.subplot(1, 2, 2)
    # Visualization of testing results.
    plt.scatter(x_test , y_test, color = '#FFCD6E', alpha = 0.7, label = 'Testing samples')
    plt.plot(x_test , linear_reg.predict(x_test), color = '#2A557F',
             label = 'Fitting curve: $y=0.2483x+16.97$')
    #plt.plot(x1_plot, x2_plot, color = 'r')
    plt.xlabel('$PM_{2.5}$ $(\mu g/m^{3})$')
    plt.ylabel('$NO_{2}$ $(\mu g/m^{3})$')
    plt.title('(b)')
    #  '$R^{2}=$' + str('{:.4f}'.format(linear_reg.score(x_test, y_test)))
    plt.legend(loc = 'upper left')
    plt.show()


def poly_regression():
    '''Poly regression.'''
    print('\nPolynomial regression:')

    # Read data and print it.
    datalist = pd.read_csv('lineardata.csv')
    X = datalist.iloc[ : , : 1].values
    Y = datalist.iloc[ : , 1].values
    # Convert to array format.
    X = np.array(datalist[['PM2.5']])
    Y = np.array(datalist['No2'])

    # Train 80% data samples.
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 1)
    poly_reg = Pipeline([
        #('std_scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree = 2)),
        ('lin_reg', linear_model.LinearRegression())
        ])
    poly_reg.fit(x_train, y_train)
    
    poly = poly_reg.get_params('poly')['poly'] 
    feature = poly.get_feature_names_out()  
    lin = poly_reg.get_params('lin_reg')['lin_reg']  
    print('\nPrint parameters of the polynomial regression model: ')
    print('Features: ', feature)
    print('Parameters: ', lin.coef_)
    print('Const: ', lin.intercept_)
    print(x_train)
    print(y_train)
    print('\nTrain Score: ', poly_reg.score(x_train, y_train))
    print('Test Score: ', poly_reg.score(x_test, y_test))

    # To calculate the coefficients.
    #poly_reg_1 = linear_model.LinearRegression()
    #poly_reg_1.fit(x_train, y_train)
    #print('\nPrint slope and intercept of the polynomial regression model: ')
    #print('Slope: ', poly_reg.coef_)
    #print('Intercept: ', poly_reg.intercept_)

    plt.style.use(['science', 'no-latex'])
    plt.figure(dpi = 165)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['mathtext.default'] = 'regular'

    # Visualization of training results.
    plt.subplot(1, 2, 1)
    plt.scatter(x_train , y_train, color = '#5FC7CA', alpha = 0.7, label = 'Training samples')
    #plt.plot(np.sort(x_train) , poly_reg.predict(np.sort(x_train)), color ='#2A557F',
    #        label = 'Fitting curve:\n $y=-0.0009679x^{2}+0.4677x+17.15$')
    x1_plot = np.linspace(0, 350)
    x2_plot = lin.coef_[2] * x1_plot ** 2 + lin.coef_[1] * x1_plot + lin.intercept_
    plt.plot(x1_plot , x2_plot, color ='#2A557F',
            label = 'Fitting curve:\n  $y=-0.0009679x^{2}+0.4677x+17.15$')
    '''
        $y=-0.0002407x^{2}+0.3838x+23.48$
        $y=-0.001099x^{2}+0.5072x+17.91$
        $y=-0.002207x^{2}+0.5623x+11.42$
        $y=-0.0009679x^{2}+0.4677x+17.15$
    '''
    plt.xlabel('$PM_{2.5}$ $(\mu g/m^{3})$')
    plt.ylabel('$NO_{2}$ $(\mu g/m^{3})$')
    #plt.title('Training set: $R^{2}=$' + str('{:.4f}'.format(poly_reg.score(x_train, y_train))))
    plt.title('(a)')
    plt.legend(loc = 'upper left')

    # Visualization of testing results.
    plt.subplot(1, 2, 2)
    plt.scatter(x_test , y_test, color = '#F3764A', alpha = 0.7, label = 'Testing samples')
    #plt.plot(np.sort(x_train) , poly_reg.predict(np.sort(x_train)), color = '#2A557F',
    #        label = 'Fitting curve:\n $y=-0.0009679x^{2}+0.4677x+17.15$')
    x1_plot = np.linspace(0, 400)
    x2_plot = lin.coef_[2] * x1_plot ** 2 + lin.coef_[1] * x1_plot + lin.intercept_
    plt.plot(x1_plot , x2_plot, color ='#2A557F',
            label = 'Fitting curve:\n  $y=-0.0009679x^{2}+0.4677x+17.15$')
    plt.xlabel('$PM_{2.5}$ $(\mu g/m^{3})$')
    plt.ylabel('$NO_{2}$ $(\mu g/m^{3})$')
    # plt.title('Test set: $R^{2}=$' + str('{:.4f}'.format(poly_reg.score(x_test, y_test))))
    plt.title('(b)')
    plt.legend(loc = 'upper left')
    plt.show()


def difference():
    datalist = pd.read_csv('lineardata.csv')
    X = datalist.iloc[ : , : 1].values
    Y = datalist.iloc[ : , 1].values
    print(datalist)
    # Convert to array format.
    X = np.array(datalist[['PM2.5']])
    Y = np.array(datalist['No2'])

    plt.style.use(['science', 'no-latex', 'grid'])
    plt.figure(dpi = 165)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['mathtext.default'] = 'regular'
    
    # 80% training samples while 20% testing samples.
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 1)
    # Introduce the training set to the linear regression model (Least Square Method).
    linear_reg = linear_model.LinearRegression()
    linear_reg.fit(x_train, y_train)

    y_pred = linear_reg.predict(x_test)
    bias = mean_squared_error(y_test, y_pred)
    print(bias)
    
    plt.plot(range(len(y_test)), y_test, '#8470FF', label='test')
    plt.plot(range(len(y_test)), y_pred, 'orange', label='predict')
    plt.legend()
    plt.show()


def Ridge():
    datalist = pd.read_csv('lineardata.csv')
    X = datalist.iloc[ : , : 1].values
    Y = datalist.iloc[ : , 1].values
    print(datalist)
    # Convert to array format.
    X = np.array(datalist[['PM2.5']])
    Y = np.array(datalist['No2'])

    plt.style.use(['science', 'no-latex', 'grid'])
    plt.figure(dpi = 165)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['mathtext.default'] = 'regular'
    
    # 80% training samples while 20% testing samples.
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 1)
    # Introduce the training set to the linear regression model (Least Square Method).
    
    linear_reg = linear_model.Ridge()
    linear_reg.fit(x_train, y_train)

    y_pred = linear_reg.predict(x_test)
    bias = mean_squared_error(y_test, y_pred)
    print(bias)
    
    print('Train Score: ', linear_reg.score(x_train, y_train))
    print('Test Score: ', linear_reg.score(x_test, y_test))

    plt.plot(range(len(y_test)), y_test, '#8470FF', label='test')
    plt.plot(range(len(y_test)), y_pred, 'orange', label='predict')
    plt.legend()
    plt.show()


#linear_regression()
print('\n=====================================\n')
Ridge()
# difference()
# poly_regression()
#linear_regression()

'''Main function'''
''' Interative interface.
    Enter '0' to test linear regression;
    Enter '1' to test poly regression;
    Enter '2' or 'Quit' or 'quit' to quit;
    Enter else will prompt and input again.
'''
#while(True):
#    print('Please choose one kind of regression：')
#    print(' [0] Linear regression')
#    print(' [1] Ploy regression')
#    print(' [2] Quit')
#    choice_num = input('Your choice is: ')

#    if (choice_num == '0'):
#        linear_regression()
#        break

#    elif (choice_num == '1'):
#        poly_regression()
#        break

#    elif (choice_num == '2' or choice_num == 'Quit' or choice_num == 'quit'):
#        break

#    else:
#        print('Input Error! Please enter the correct format!\n')

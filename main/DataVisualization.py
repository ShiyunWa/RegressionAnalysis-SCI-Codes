import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def dataset_visual():
    PM_25 = pd.read_csv('lineardata.csv', usecols = [0])
    NO_2 = pd.read_csv('lineardata.csv', usecols = [1])
    PM_25.describe()
    NO_2.describe()

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    years = ['2017', '2018', '2019', '2020', '2021']

    plt.style.use(['science', 'no-latex'])
    plt.figure(dpi = 165)
    #plt.rcParams['font.family'] = 'sans-serif'
    #plt.rcParams['font.sans-serif'] = 'Arial'
    #plt.rcParams.update({'font.size': 8})
    #plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['mathtext.default'] = 'regular'

    plt.subplot(1, 2, 1)
    plt.plot(PM_25, color = '#2A557F', label = '$PM_{2.5}$ concentration')
    plt.grid(visible = True, which = 'major', linestyle = '-', color = '#C0C0C0')
    plt.grid(visible = True, which = 'minor', linestyle = '--', color = '#C0C0C0', alpha = 0.5)
    plt.xlabel('Date in five years')
    plt.ylabel('$PM_{2.5}$ $(\mu g/m^{3})$')
    # plt.xticks(range(1, 366, 31), months, rotation = 45)
    plt.xticks(range(1, 1825, 366), years, rotation = 45)
    plt.title('(a)')
    plt.legend(loc = 'upper right')

    plt.subplot(1, 2, 2)
    plt.plot(NO_2, color = '#87CEEB', label = '$NO_{2}$ concentration')
    plt.grid(visible = True, which = 'major', linestyle = '-', color = '#C0C0C0')
    plt.grid(visible = True, which = 'minor', linestyle = '--', color = '#C0C0C0', alpha = 0.5)
    plt.xlabel('Date in five years')
    plt.ylabel('$NO_{2}$ $(\mu g/m^{3})$')
    # plt.xticks(range(1, 366, 31), months, rotation = 45)
    plt.xticks(range(1, 1825, 366), years, rotation = 45)
    plt.legend(loc = 'upper right')
    plt.title('(b)')
    plt.show()


def poly_visual():
    plt.style.use(['science', 'no-latex'])
    plt.figure(dpi = 160)
    plt.rcParams['font.family'] = "Times New Roman"
    plt.rcParams.update({'font.size': 9})

    training_2017 = ['0.6246', '0.6249', '0.6352', '0.6360'] 
    test_2017 = ['-0.02796', '-0.04249', '-0.2601', '-0.1906']
    training_2017_2021 = ['0.4799', '0.4807', '0.4937', '0.4937']
    test_2017_2021 = ['0.4131', '0.4672', '-0.6248', '-0.9717']
    degrees = ['2-degree', '3-degree', '4-degree', '5-degree']

    plt.subplot(1, 2, 1)
    plt.plot(training_2017, color = '#2A557F', label = '2017 training set',
             marker = '.', markersize = 4)
    plt.plot(training_2017_2021, color = '#2A557F', label = '2017-2021 training set',
             marker = '.', markersize = 4)
    plt.grid(visible = True, which = 'major', linestyle = '-', color = '#C0C0C0')
    plt.grid(visible = True, which = 'minor', linestyle = '--', color = '#C0C0C0', alpha = 0.5)
    plt.title('The fluctuation of $R^{2}$ in each training set')
    plt.xlabel('Degree')
    plt.ylabel('$R^{2}$')
    plt.xticks(range(0, 4), degrees, rotation = 45)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_2017, color = '#87CEEB', label = '2017 test set',
             marker = '.', markersize = 4)
    plt.plot(test_2017_2021, color = '#87CEEB', label = '2017-2021 test set',
             marker = '.', markersize = 4)
    plt.grid(visible = True, which = 'major', linestyle = '-', color = '#C0C0C0')
    plt.grid(visible = True, which = 'minor', linestyle = '--', color = '#C0C0C0', alpha = 0.5)
    plt.title('The fluctuation of $R^{2}$ in each test set')
    plt.xlabel('Degree')
    plt.ylabel('$R^{2}$')
    plt.xticks(range(0, 4), degrees, rotation = 45)
    plt.legend()
    plt.show()

dataset_visual()
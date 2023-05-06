import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn import metrics


#Pre_processing.py File
def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X

#Read Data
df = pd.read_csv('SuperMarketSales.csv')

#Clean Data
df.dropna(how='any', inplace=True)








# <editor-fold desc="Date Features">
sellDate = df['Date']
def DatesFeatures(Var):
    DAtes = []
    if Var =='Month':
        for date in sellDate:
            try:
                converted_dates = datetime.strptime(date, '%d/%m/%Y').date()
                DAtes.append(converted_dates.month)
            except ValueError as ve:
                converted_dates = datetime.strptime(date, '%d-%m-%Y').date()
                DAtes.append(converted_dates.month)
        Dates2 = pd.Series(pd.arrays.SparseArray(DAtes))


    elif Var == 'Year':
        for date in sellDate:
            try:
                converted_dates = datetime.strptime(date, '%d/%m/%Y').date()
                DAtes.append(converted_dates.year)
            except ValueError as ve:
                converted_dates = datetime.strptime(date, '%d-%m-%Y').date()
                DAtes.append(converted_dates.year)
        Dates2 = pd.Series(pd.arrays.SparseArray(DAtes))


    elif Var == 'Day':
        for date in sellDate:
            try:
                converted_dates = datetime.strptime(date, '%d/%m/%Y').date()
                DAtes.append(converted_dates.day)
            except ValueError as ve:
                converted_dates = datetime.strptime(date, '%d-%m-%Y').date()
                DAtes.append(converted_dates.day)
        Dates2 = pd.Series(pd.arrays.SparseArray(DAtes))

    return Dates2

day = DatesFeatures('Day')
month = DatesFeatures('Month')
year = DatesFeatures('Year')

df2 = df.drop(columns=['Date'])

df2.insert(loc=1, column='Day',value=day)
df2.insert(loc=2, column='Month',value=month)
df2.insert(loc=3, column='Year',value=year)
# </editor-fold>

# X = df2.iloc[:,4:7]
# Y = df2['Weekly_Sales']
# #Feature Selection
# #Get the correlation between the features
# corr = df2.corr()
# #Top 50% Correlation training features with the Value
# top_feature = corr.index[abs(corr['Weekly_Sales'])>0.5]
# #Correlation plot
# plt.subplots(figsize=(12, 8))
# top_corr = df2[top_feature].corr()
# sns.heatmap(top_corr, annot=True)
# plt.show()
# top_feature = top_feature.delete(-1)
# X = X[top_feature]


#Features Assignment
x = df2







# <editor-fold desc="Features Standarization (Temperature, Feuel_Price, CPI)">
for i in range(4,x.shape[1]):
    tmpcol = []
    tmp = x.iloc[:,i]
    m = np.mean(tmp)
    s = np.std(tmp.to_numpy())
    for j in range(len(tmp)):
        #print('j = ', j)
        tmpcol.append((tmp[j]-m)/s)
    standX = pd.Series(pd.arrays.SparseArray(tmpcol))
    colname = x.columns[i]
    x.drop(columns = colname, axis =1, inplace =True)
    x.insert(loc=i, column = 'STAND_'+colname, value = standX)
StandrizedData = x
# </editor-fold>
y = StandrizedData['STAND_Weekly_Sales']
StandrizedData.drop(columns=['STAND_Weekly_Sales'], inplace=True)
# print(StandrizedData)
# print(y)

def Poly_Trans(x, degree):
    arrOfPolly = np.zeros(shape=[x.shape[0], degree + 1])

    for i in range(degree + 1):
        arrOfPolly[:, i] = x ** i
    return arrOfPolly

#Polynomial Regression Function
def Poly_Regression(x, y, degree, train):


    arrOfPolly =  Poly_Trans(x, degree)
    # arrOfPolly = np.zeros(shape=[x.shape[0],degree+1])
    #
    # for i in range(degree+1):
    #     arrOfPolly[:,i] = x**i
    #print(arrOfPolly)



    #Fitting Model
    FitVar = LinearRegression()
    FitVar.fit(arrOfPolly, y)
    y_pred = FitVar.predict(arrOfPolly)

    #Print Coeffeicents
    v = 'Test'
    if(train == True):
        v = 'Train'
    print(f'\n\t\t !!!!!!!!!! {x.name} {v} Data !!!!!!!!!!\t\t\n ')
    print('Y_Pred = ',y_pred)
    print("Function Coefficent: \n", FitVar.coef_)
    print("Function Intercept: \n", FitVar.intercept_)
    print("Mean SQR Error = ", metrics.mean_squared_error(y, y_pred))

    #Plot the gragh
    plt.scatter(x,y, marker='x', c='b')
    plt.plot(x,y_pred,c='r')
    plt.title(f'{x.name} {v} Data')
    plt.xlabel(f'{x.name} {v} Set')
    plt.ylabel(f'Weekly Sales {v} Set')
    plt.show()



def Run_spilt_data(x,y, dgree):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, shuffle=True)
    print(f'\n\n\n\t\t ############# {x.name} #############\t\t\n')
    #Train Data
    #print("\n\n\t\t************ Trainning Data ************\t\n\n")
    Poly_Regression(x_train,y_train, dgree, True)
    #Test Data
    #print("\t\t\n\n************ Test Data ************\t\n\n")
    #Poly_Regression(x_test,y_test,dgree, False)
    x_poly_test = Poly_Trans(x_test, dgree)
    LReg = LinearRegression()
    LReg.fit(x_poly_test,y_test)
    y_test_Pred = LReg.predict(x_poly_test)
    print('\ny_test_pred = \n ', y_test_Pred)




#Main
for i in range(StandrizedData.shape[1]):
    Run_spilt_data(StandrizedData.iloc[:,i], y, 4)





































import random
import datetime
import math
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import csv

#　平日日付リストを取得
def getDateList(sDate, eDate):
    dCnt = (eDate-sDate).days
    dList = []
    for i in range(dCnt+1):
        tmpDate = sDate+datetime.timedelta(days=i)
        if tmpDate.weekday() <= 4 :
            dList.append(tmpDate)
    return dList, dCnt

#　リストに無い乱数を生成
def getDistinctRand(rList, minVal, maxVal):
    r = random.randint(minVal, maxVal)
    if rList.count(r) > 0:
        r = getDistinctRand(rList, minVal, maxVal)
    return r

#　Volatilityの転換期を取得
def getX(dCnt, turningCnt):
    x = []
    for i in range(turningCnt):
        x.append(getDistinctRand(x, i*round(dCnt/turningCnt, 0), (i+1)*round(dCnt/turningCnt, 0)))
    x.append(1)
    x.append(dCnt)
    x.sort()
    return x

#　転換期のVolatilityを取得
def getY(rng, minVal, maxVal):
    y = []
    for i in range(rng):
        tmpY = random.uniform(minVal, maxVal)
        y.append(tmpY)
    return y

#　指数データを取得
def getCumulativeRtn(rtn):
    cumRtn = [10000]
    for i in range(len(rtn)):
        cumRtn.append(cumRtn[-1]*(1+rtn[i]))
    return cumRtn

#　先物倍率系列を取得
def getFRatio(index, rtn, dList, baseFRatio):
    fRatio = []
    for i in range(len(rtn)):
        if i == 0:
            baseVal = index[i]
            fRatio.append(baseFRatio)                
        elif i == len(rtn)-1 or dList[i].month == dList[i+1].month:
            fRatio.append((index[i]/baseVal)*baseFRatio/(1+baseFRatio*(index[i]/baseVal-1)))
        elif dList[i].month != dList[i+1].month:
            baseVal = index[i]
            fRatio.append(baseFRatio)
    return fRatio
    
##　スプライン補間
#def spline(x, y, point, deg):
#    tck,u = interpolate.splprep([x,y], k=deg, s=0)
#    u = np.linspace(0, 1, num=point, endpoint=True)
#    spline = interpolate.splev(u, tck)
#    return spline[0],spline[1]

#　秋間スプライン補間
def akimaSpline(x, y, point):
    f = interpolate.Akima1DInterpolator(x, y)
    X = np.linspace(x[0], x[-1], num=point, endpoint=True)
    Y = f(X)
    return X, Y

def makeVolaData(sDate, eDate, volaPeriod, minVola, maxVola):
    dList, dCnt = getDateList(sDate, eDate)
    turningCnt = int(round(len(dList)/volaPeriod, 0))
    x = getX(dCnt, turningCnt)
    y = getY(len(x), minVola, maxVola)
#    X, Y = spline(x, y, dCnt, 3)
    X, Y = akimaSpline(x, y, len(dList))
    if min(Y) <= 0.05/math.sqrt(260) or max(Y) >= 0.4/math.sqrt(260) or np.isnan(Y).any():
        x, y, X, Y, dList = makeVolaData(sDate, eDate, volaPeriod, minVola, maxVola)
    return x, y, X, Y, dList

def plotOnGraph(x, y1, y2, y3):
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.xlim([min(x), max(x)])
    plt.ylim([min(y2), max(y2)])
    plt.legend(loc='lower right')
    plt.grid(which='major',color='black',linestyle='-')
    plt.grid(which='minor',color='black',linestyle='-')
#    plt.xticks(list(filter(lambda x: x%1==0, np.arange(0, turningCnt))))
#    plt.yticks(list(filter(lambda x: x%1==0, np.arange(minVola, maxVola))))
    plt.show()

def plot(x, y, xName, yName, title):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.scatter(x,y)
    ax.set_title(title)
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)

def mainProc(loopCnt):
    sDate = datetime.date(2009, 9, 30)
    eDate = datetime.date(2019, 9, 30)
    volaPeriod = 260
    minVola = 0.10/math.sqrt(260)
    maxVola = 0.25/math.sqrt(260)
    minRR = -0.5
    maxRR = 1
    baseFRatio = 1.8
    
    indexRtn = []
    dFundRtn = []
    mFundRtn = []
    winFlg = []
    loseFlg = []
    for i in range(loopCnt):
        xRaw, volaRaw, x, vola, dList = makeVolaData(sDate, eDate, volaPeriod, minVola, maxVola)
#        xRaw, volaRaw, x, RR, dList = makeVolaData(sDate, eDate, volaPeriod, minRR, maxRR)
#        aveRtn = [vola[j]*RR[j]/math.sqrt(260) for j in range(len(vola))]
        aveRtn = [v*random.uniform(minRR, maxRR)/math.sqrt(260) for v in vola]
        rtn = [np.random.normal(aveRtn[j], vola[j]) for j in range(len(vola)-1)]
        index = getCumulativeRtn(rtn)
        fRatio = getFRatio(index, rtn, dList, baseFRatio)
        dFund = getCumulativeRtn([baseFRatio*rtn[j] for j in range(len(rtn))])
        mFund = getCumulativeRtn([fRatio[j]*rtn[j] for j in range(len(rtn))])
        indexRtn.append(round(index[-1]/index[0]-1, 2))
        dFundRtn.append(dFund[-1]/dFund[0]-1)
        mFundRtn.append(mFund[-1]/mFund[0]-1)
        if mFundRtn[-1]-dFundRtn[-1] > 0:
            winFlg.append(1)
            loseFlg.append(0)
        else:
            winFlg.append(0)
            loseFlg.append(1)
        if i%1000 == 0:
            print(i)

    df1 = pd.DataFrame({'indexRtn' : indexRtn})
    df2 = pd.DataFrame({'dFundRtn' : dFundRtn})
    df3 = pd.DataFrame({'mFundRtn' : mFundRtn})
    df4 = pd.DataFrame({'winFlg' : winFlg})
    df5 = pd.DataFrame({'loseFlg' : loseFlg})
    dfRaw = pd.concat([df1, df2, df3, df4, df5], axis=1)
    dfAgg = dfRaw.groupby('indexRtn').sum()

#    df1 = pd.DataFrame({'index' : index})
#    df2 = pd.DataFrame({'dFund' : dFund})
#    df3 = pd.DataFrame({'mFund' : mFund})
#    df = pd.concat([df1, df2, df3], axis=1)
#    print(df)
#    print(index)
#    print(dFund)
#    print(mFund)
    dfAgg.to_csv('result_' + str(loopCnt) + 'data.csv')

#    plotOnGraph(xRaw, volaRaw, volaRaw, volaRaw)
#    plotOnGraph(x, aveRtn, aveRtn, aveRtn)
#    plotOnGraph(x, index, index, index)

start = time.time()
mainProc(2000000)
elapsed_time = time.time()-start
print("elapsed_time:{0}".format(elapsed_time)+"[sec]")

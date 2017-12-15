from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from operator import truediv
import numpy as np
from anfis import membership
from anfis import anfis
from sklearn.metrics import accuracy_score

dataset = load_iris()
x = dataset.data
y = dataset.target

# sizecv = 5
# kf = StratifiedKFold(n_splits=sizecv, shuffle=True, random_state=123)
# for train, test in kf.split(X, Y):
df = pd.DataFrame(dataset.data)
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
y_test = y_test.tolist()
print y_test
# x_train = x[:100]
# y_train = y[:100]
# x_test = x[100:]
# y_test = y[100:]
# print y_test
# for i in range(len(y_test)):
#     y_actual.append(y_test[i])
# x_train = X[train]
# y_train = Y[train]
# x_test = X[test]
# y_test = Y[test]
# print x_train
# print y_train

mf = [[['gaussmf', {'mean': 4.6, 'sigma': 4.6}], ['gaussmf', {'mean': 5.1, 'sigma': 5.1}],
       ['gaussmf', {'mean': 6.5, 'sigma': 6.1}]],
      [['gaussmf', {'mean': 3.4, 'sigma': 3.1}], ['gaussmf', {'mean': 4.0, 'sigma': 4.0}],
       ['gaussmf', {'mean': 5.1, 'sigma': 5.1}]],
      [['gaussmf', {'mean': 1.4, 'sigma': 1.2}], ['gaussmf', {'mean': 4.0, 'sigma': 3.1}],
       ['gaussmf', {'mean': 2.2, 'sigma': 2.1}]],
      [['gaussmf', {'mean': 0.3, 'sigma': 0.2}], ['gaussmf', {'mean': 1.0, 'sigma': 1.1}],
       ['gaussmf', {'mean': 2.4, 'sigma': 2.4}]]]
mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(x_train, y_train, mfc)
anf.trainHybridJangOffLine(epochs=3)
# for i in range(0,119):
#      print round(anf.fittedValues[i],3)
y_predicted = []
for i in range(len(y_test)):
    res = round(anf.fittedValues[y_test[i]],3)
    print res
    if abs(res-0) < abs(res -1) < abs(res -2):
        y_predicted.append(0)
    elif abs(res-0) > abs(res -1) < abs(res -2):
        y_predicted.append(1)
    elif abs(res-0) > abs(res-1) > abs(res-2):
        y_predicted.append(2)

trupred = 0
print y_test
print y_predicted
# print accuracy_score(y_test, y_predicted)*100, "%"
#check accuracy
for i in range(len(y_predicted)):
    if y_predicted[i] == y_test[i]:
        trupred +=1
print truediv(trupred,50)*100, "%"
# anf.plotErrors()
# anf.plotMF(12, 30)
anf.plotResults()
# print round(anf.consequents[-1][0], 6)
# print round(anf.consequents[-2][0], 6)
# print round(anf.fittedValues[9][0], 6)
# if round(anf.consequents[-1][0], 6) == -5.275538 and round(anf.consequents[-2][0], 6) == -1.990703 and round(
#         anf.fittedValues[9][0], 6) == 0.002249:
#     print 'test is good'
# anf.plotErrors()
# anf.plotResults()
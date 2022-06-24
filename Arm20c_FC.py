import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

nc = 5      # number of classes
cn = [1, 2, 3, 4, 5, 6, 7, 8]  # animal id

# load data
cf = pd.read_csv('C:/Data/animal activity/Arm20c_Acc_GNSS_features.csv')

it = 100  # number of trials
mc, mcc, cr = np.zeros(it), np.zeros([nc, it]), np.zeros([it, nc, nc])

for rs in range(it):
    tstya, prdya = [], []

    for tt in range(len(cn)):
        # training data
        trnx = cf[cf.animal!=cn[tt]][['mx', 'my', 'mz', 'sx', 'sy', 'sz', 'speed', 'DtWP', 'error']].values
        trny = cf[cf.animal!=cn[tt]].label.values

        # test data
        tstx = cf[cf.animal==cn[tt]][['mx', 'my', 'mz', 'sx', 'sy', 'sz', 'speed', 'DtWP', 'error']].values
        tsty = cf[cf.animal==cn[tt]].label.values

        # standardize the data
        scaler = StandardScaler().fit(trnx)
        trnx, tstx = scaler.transform(trnx), scaler.transform(tstx)

        # train the model
        mdl = MLPClassifier(hidden_layer_sizes=(7,), alpha=1e-4, activation='relu', solver='lbfgs', max_iter=10000)\
            .fit(trnx, trny)

        tstya = np.append(tstya, tsty)
        prdya = np.append(prdya, mdl.predict(tstx))                 # predicted classes

    cr[rs] = confusion_matrix(tstya, prdya, normalize=None)
    mc[rs] = matthews_corrcoef(tstya, prdya)
    for ii in range(nc):
        tp = cr[rs][ii, ii]
        fp = np.sum(cr[rs][:, ii]) - cr[rs][ii, ii]
        tn = np.sum(cr[rs]) - np.sum(cr[rs][:, ii]) - np.sum(cr[rs][ii, :]) + cr[rs][ii, ii]
        fn = np.sum(cr[rs][ii, :]) - cr[rs][ii, ii]
        pr = tp/(tp+fp)
        re = tp/(tp+fn)
        mcc[ii, rs] = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn))/np.sqrt((tn+fp)*(tn+fn))

# print results
c = np.mean(cr, axis=0)
np.set_printoptions(formatter={'float': lambda x: '{0:0.0f}'.format(x)})
print(c)
for ii in range(len(c)):
    print('MCC{0}: {1:0.4f} ± {2:.4f}'.format(ii, np.mean(mcc[ii]), np.std(mcc[ii])))
print('MCC:  {0:0.4f} ± {1:.4f}'.format(np.mean(mc), np.std(mc)))

plt.show()

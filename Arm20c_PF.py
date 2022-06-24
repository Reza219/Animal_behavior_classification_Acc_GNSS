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
pp = cf.label.value_counts(normalize=True).sort_index().values  # prior probabilities

it = 100  # number of trials
mc, mcc, cr = np.zeros(it), np.zeros([nc, it]), np.zeros([it, nc, nc])

for rs in range(it):
    tstya, pra = [], np.empty([0, nc])

    for tt in range(len(cn)):
        # training data
        trnxi = cf[cf.animal!=cn[tt]][['mx', 'my', 'mz', 'sx', 'sy', 'sz']].values
        trnxg = cf[cf.animal!=cn[tt]][['speed', 'DtWP', 'error']].values
        trny = cf[cf.animal!=cn[tt]].label.values

        # test data
        tstxi = cf[cf.animal==cn[tt]][['mx', 'my', 'mz', 'sx', 'sy', 'sz']].values
        tstxg = cf[cf.animal==cn[tt]][['speed', 'DtWP', 'error']].values
        tsty = cf[cf.animal==cn[tt]].label.values

        # standardize the data
        sci = StandardScaler().fit(trnxi)
        trnxi, tstxi = sci.transform(trnxi), sci.transform(tstxi)
        scg = StandardScaler().fit(trnxg)
        trnxg, tstxg = scg.transform(trnxg), scg.transform(tstxg)

        # train the model
        mdli = MLPClassifier(hidden_layer_sizes=(5,), alpha=1e-4, activation='relu', solver='lbfgs', max_iter=10000)\
            .fit(trnxi, trny)
        mdlg = MLPClassifier(hidden_layer_sizes=(4,), alpha=1, activation='relu', solver='lbfgs', max_iter=10000)\
            .fit(trnxg, trny)

        tstya = np.append(tstya, tsty)
        pri, prg = mdli.predict_proba(tstxi), mdlg.predict_proba(tstxg)
        prf = pri*prg/pp
        pra = np.append(pra, prf/np.sum(prf, axis=1).reshape(-1, 1), axis=0)  # predicted class probabilities

    prdya = np.argmax(pra, axis=1)
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

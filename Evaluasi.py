# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:39:49 2019

@author: vickzjr
"""
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.ion()

training_predicted = dnn.predict(train_data)
testing_predicted = dnn.predict(test_data)
from sklearn.metrics import confusion_matrix
cm_training = confusion_matrix(train_label.argmax(axis=-1),training_predicted.argmax(axis=-1))
cm_training = np.array(cm_training)
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/confusion matrix train {0}.csv'.format(model),cm_training,fmt='%d',delimiter=',')

cm_testing = confusion_matrix(test_label.argmax(axis=1),testing_predicted.argmax(axis=1))
cm_testing = np.array(cm_testing)
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/confusion matrix test model {0}.csv'.format(model),cm_testing,fmt='%d',delimiter=',')

#Testing

Sen_Class = []
Spe_Class = []
Pre_Class = []
F1_Class = []
Err_Class = []
Acc_Class = []
FP_Class = []
FN_Class = []
TP_Class = []
TN_Class = []

for idx in range(len(cm_training)):
        TP = cm_training[idx, idx]
        FN = np.sum(cm_training[idx, :]) - TP
        FP = np.sum(cm_training[:, idx]) - TP
        TN = np.sum(cm_training) - (TP + FN + FP)
        
        Sen = TP / (TP + FN)
        Spe = TN / (TN + FP)
        Pre = TP / (TP + FP)
        F1 = (2 * Pre * Sen) / (Sen + Pre)
        Err = (FP + FN) / (FP + FN + TN + TP)
        Acc = (TP + TN) / (FP + FN + TN + TP)
        
        Sen_Class.append([Sen, idx])
        Spe_Class.append([Spe, idx])
        Pre_Class.append([Pre, idx])
        F1_Class.append([F1, idx])
        Err_Class.append([Err, idx])
        Acc_Class.append([Acc, idx])
        FP_Class.append([FP,idx])
        FN_Class.append([FN,idx])
        TP_Class.append([TP,idx])
        TN_Class.append([TN,idx])


Sen_Class_numpy = np.array(Sen_Class)
avg_sen = np.mean(Sen_Class_numpy[:,0])
Sen_Class.append([avg_sen,10])
Sen_Class = np.transpose(Sen_Class)
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/sensitivity cm training {0}.csv'.format(model),Sen_Class,fmt='%0.4f',delimiter=',')

Spe_Class_numpy = np.array(Spe_Class)
avg_spe = np.mean(Spe_Class_numpy[:,0])
Spe_Class.append([avg_spe,10])
Spe_Class = np.transpose(Spe_Class)
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/specificity cm training {0}.csv'.format(model),Spe_Class,fmt='%0.4f',delimiter=',')

Pre_Class_numpy = np.array(Pre_Class)
avg_pre = np.mean(Pre_Class_numpy[:,0])
Pre_Class.append([avg_pre,10])
Pre_Class = np.transpose(Pre_Class)
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/precision cm training {0}.csv'.format(model),Pre_Class,fmt='%0.4f',delimiter=',')

F1_Class_numpy = np.array(F1_Class)
avg_f1 = np.mean(F1_Class_numpy[:,0])
F1_Class.append([avg_f1,10])
F1_Class = np.transpose(F1_Class)
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/F1-score cm training {0}.csv'.format(model),F1_Class,fmt='%0.4f',delimiter=',')

Acc_Class_numpy = np.array(Acc_Class)
avg_acc = np.mean(Acc_Class_numpy[:,0])
Acc_Class.append([avg_acc,10])
Acc_Class = np.transpose(Acc_Class)
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/accuracy cm training {0}.csv'.format(model),Acc_Class,fmt='%0.4f',delimiter=',')

np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/TP cm training {0}.csv'.format(model),TP_Class,fmt='%d',delimiter=',')
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/FP cm training {0}.csv'.format(model),FP_Class,fmt='%d',delimiter=',')
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/FN cm training {0}.csv'.format(model),FN_Class,fmt='%d',delimiter=',')
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/TN cm training {0}.csv'.format(model),TN_Class,fmt='%d',delimiter=',')


#Testing
Sen_Class = []
Spe_Class = []
Pre_Class = []
F1_Class = []
Err_Class = []
Acc_Class = []
FP_Class = []
FN_Class = []
TP_Class = []
TN_Class = []

for idx in range(len(cm_testing)):
        TP = cm_testing[idx, idx]
        FN = np.sum(cm_testing[idx, :]) - TP
        FP = np.sum(cm_testing[:, idx]) - TP
        TN = np.sum(cm_testing) - (TP + FN + FP)
        
        Sen = TP / (TP + FN)
        Spe = TN / (TN + FP)
        Pre = TP / (TP + FP)
        F1 = (2 * Pre * Sen) / (Sen + Pre)
        Err = (FP + FN) / (FP + FN + TN + TP)
        Acc = (TP + TN) / (FP + FN + TN + TP)
        
        Sen_Class.append([Sen, idx])
        Spe_Class.append([Spe, idx])
        Pre_Class.append([Pre, idx])
        F1_Class.append([F1, idx])
        Err_Class.append([Err, idx])
        Acc_Class.append([Acc, idx])
        FP_Class.append([FP,idx])
        FN_Class.append([FN,idx])
        TP_Class.append([TP,idx])
        TN_Class.append([TN,idx])


Sen_Class_numpy = np.array(Sen_Class)
avg_sen = np.mean(Sen_Class_numpy[:,0])
Sen_Class.append([avg_sen,10])
Sen_Class = np.transpose(Sen_Class)
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/sensitivity cm testing {0}.csv'.format(model),Sen_Class,fmt='%0.4f',delimiter=',')

Spe_Class_numpy = np.array(Spe_Class)
avg_spe = np.mean(Spe_Class_numpy[:,0])
Spe_Class.append([avg_spe,10])
Spe_Class = np.transpose(Spe_Class)
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/specificity cm testing {0}.csv'.format(model),Spe_Class,fmt='%0.4f',delimiter=',')

Pre_Class_numpy = np.array(Pre_Class)
avg_pre = np.mean(Pre_Class_numpy[:,0])
Pre_Class.append([avg_pre,10])
Pre_Class = np.transpose(Pre_Class)
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/precision cm testing {0}.csv'.format(model),Pre_Class,fmt='%0.4f',delimiter=',')

F1_Class_numpy = np.array(F1_Class)
avg_f1 = np.mean(F1_Class_numpy[:,0])
F1_Class.append([avg_f1,10])
F1_Class = np.transpose(F1_Class)
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/F1-score cm testing {0}.csv'.format(model),F1_Class,fmt='%0.4f',delimiter=',')

Acc_Class_numpy = np.array(Acc_Class)
avg_acc = np.mean(Acc_Class_numpy[:,0])
Acc_Class.append([avg_acc,10])
Acc_Class = np.transpose(Acc_Class)
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/accuracy cm testing {0}.csv'.format(model),Acc_Class,fmt='%0.4f',delimiter=',')

np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/TP cm testing {0}.csv'.format(model),TP_Class,fmt='%d',delimiter=',')
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/FP cm testing {0}.csv'.format(model),FP_Class,fmt='%d',delimiter=',')
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/FN cm testing {0}.csv'.format(model),FN_Class,fmt='%d',delimiter=',')
np.savetxt('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/TN cm testing {0}.csv'.format(model),TN_Class,fmt='%d',delimiter=',')

###############################
#ROC CURVE
# Compute macro-average ROC curve and ROC area
from scipy import interp
from sklearn.metrics import roc_curve, auc
from itertools import cycle
n_classes = 3

#Inisialisasi 
fpr = dict()
tpr = dict()
roc_auc = dict()
lw = 2
for i in np.arange(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_label[:, i], testing_predicted[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in np.arange(3):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(20,20))

#colors = cycle(['aqua', 'darkorange', 'black','red','green','pink','purple','yellow','brown','grey'])
colors = cycle(['aqua', 'darkorange', 'black'])
for i, color in zip(np.arange(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve 3 class')
plt.legend(loc="lower right")
plt.savefig('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/ROC curve {0}.jpg'.format(model))
plt.close()

#Precision Recall Curve

from itertools import cycle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in np.arange(3):
    precision[i], recall[i], _ = precision_recall_curve(test_label[:, i],
                                                        testing_predicted[:, i])
    average_precision[i] = average_precision_score(test_label[:, i], testing_predicted[:, i])

    
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange','yellow'])

plt.figure(figsize=(20, 20))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
#l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
#labels.append('micro-average Precision-recall (area = {0:0.2f})'
        #      ''.format(average_precision["micro"]))

for i, color in zip(np.arange(3), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve 4 class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

plt.savefig('F:/Aritmia/new-aritmia/Data_hasil/Train dan test/PR curve {0}.jpg'.format(model))
plt.close()

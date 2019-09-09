import wfdb
import numpy as np
from wfdb.processing import normalize_bound
import utility

path = 'F:/Aritmia/new-aritmia/Dataset/SVDB/*.atr'

path_split = utility.glob_items(path)

list_beats = []
list_labels = []
list_another_labels = []
for item in np.arange(len(path_split)):
    record = wfdb.rdrecord(path_split[item])
    record_dict = record.__dict__
    signal = record_dict['p_signal'][:,0]
    annotation = wfdb.rdann(path_split[item],'atr')
    ann_dict = annotation.__dict__
    symbol = ann_dict['symbol']
    peaks = ann_dict['sample']
    name = ann_dict['record_name']
    fs = ann_dict['fs']
    
    t1 = np.int(0.25 * fs)
    t2 = np.int(0.45 * fs)
    peak = np.arange(len(peaks))
    
    new_signal = utility.wavelet_transform(signal,8,'sym5')
    new_signal = normalize_bound(new_signal)
    
    signal = normalize_bound(signal)
    
    beats = []
    labels = []
    for x in peak:
        if (peaks[x] - t1) > 0 and (peaks[x] + t2) < len(signal):
            #if (symbol[x] == 'A' or symbol[x] == 'L' or symbol[x] == 'N' or symbol[x] == '.' or symbol[x] == 'P' or symbol[x] == 'R' or symbol[x] == 'V' or symbol[x] == 'f' or symbol[x] == 'F' or symbol[x] == '!' or symbol[x] == 'j' ):
            if (symbol[x] == '~' or symbol[x] == '|' or symbol[x] == '+' or symbol[x] == 'B' or symbol[x] == 'F' or symbol[x] == 'f' or symbol[x] == 'Q' or symbol[x] == 'a' or symbol[x] == 'J'):
                continue
            else:

                beat = signal[peaks[x] - t1 : peaks[x] + t2 ]
                beats.append(beat)
                symbol[x] = check_label(symbol[x])             
                labels.append(symbol[x])
                #print('Saving beats {0} in progress countin {1}'.format(name,x))
                #print('Saving label {0} in progress counting {1}'.format(name,x))
    #print('Saving started')
    #np.savetxt('F:/Aritmia/new-aritmia/Dataset/SVDB/Beats SVDB non bwr {0}.csv'.format(name),beats,delimiter=',',fmt='%.3f')
    #np.savetxt('F:/Aritmia/new-aritmia/Dataset/SVDB/Labels SVDB non bwr {0}.csv'.format(name),labels,fmt='%s')
    print('Saving beats {0} done'.format(name))
    print('Saving labels {0} done'.format(name))
    list_beats.extend(beats)
    list_labels.extend(labels)
    list_another_labels.extend(another_labels)


list_beats = np.array(list_beats)
list_labels = np.array(list_labels)
#np.savetxt('F:/Aritmia/new-aritmia/Dataset/SVDB/Beats SVDB 3 class.csv',list_beats,delimiter=',',fmt='%.3f')
#np.savetxt('F:/Aritmia/new-aritmia/Dataset/SVDB/Labels SVDB non bwr 5 class.csv',list_labels,fmt='%s')

import os

import pandas as pd
from Model import *
import matplotlib.pyplot as plt
from Data.data import *
from Conform import Conformer
from RNNformer import RNNformer
from FEA import FEA_T

plt.rcParams['font.sans-serif'] = ['SimHei']
#model = FEA_T(device='cpu')
modelname = 'Conformer12_2_CR=2,norescov'
model = Conformer(Device='cpu', depth=12, trans2con=True, channel_ratio=2)
#model = RNNformer(data_length=128, in_chans=2,  channel_ratio=2, depth=9, base_channel=2, num_heads=4, mlp_ratio=4, qkv_bias=True, Device='cpu', drop_rate=0.5, drop_path_rate=0.5)
model_param = f'models/models_{modelname}/{modelname}.pt'
csv_path = f'save_csvdata/{modelname}.csv'
csv_acc = f'save_csvdata/{modelname}acc.csv'
img_path = f'sava_img/{modelname}'
os.makedirs('save_csvdata', exist_ok=True)
os.makedirs(img_path,exist_ok=True)
#model_param = 'models/models_RNNFormerAP/RNNFormerAP.pt'
#model_param = './models_TransFormer/TransFormer.pt'
if model_param:
    model.load_state_dict(torch.load(model_param))
    print(f'load{model_param}')
model.eval()
def plot_all_SNR(acc):
    plt.plot(sorted(SNRS), map(lambda x: acc[x], sorted(SNRS)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10 Alpha")

def plot_confusion_matrix(cm, labels=[], title='Confusion matrix', cmap=plt.cm.Blues, SNR=-20):
    for first_index in range(len(cm)):  # 第几行
        for second_index in range(len(cm[first_index])):  # 第几列
            temp = cm[first_index][second_index]*100
            if temp == 0.0 or temp == 100.0:
                plt.text(first_index, second_index, int(temp), va='center',
                         ha='center',
                         fontsize=8)
            else:
                plt.text(first_index, second_index, r'{0:.2f}'.format(temp), va='center',
                         ha='center',
                         fontsize=8)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=8)
    plt.xlabel('Predicted label', fontsize=8)
    plt.savefig(f'{img_path}/{SNR}.png')
    plt.show()

n_train = 220000 * 0.3
np.random.seed(2016)
testidx = sorted(np.random.choice(range(0,220000), size=int(n_train), replace=False))
sub_Acc = pd.DataFrame(columns=MODULATIONS, index=sorted(SNRS))
acc = {}
for snr in sorted(SNRS):
    # extract classes @ SNR
    snrid = [a for a in testidx if dataset.y[a][1] == snr and dataset.y[a][0] == 10]
    Testx = []
    for i in snrid:
        Testx.append(torch.from_numpy(dataset.X[i]).to(torch.float).unsqueeze(0))
    Testx_tensor = torch.cat(Testx, dim=0).unsqueeze(1)
    Testy = []
    for i in snrid:
        Testy.append(MODULATIONS[dataset.y[i][0]])

    # estimate classes
    feature = model(Testx_tensor)
    for i in range(0,Testx_tensor.shape[0]):
        j = Testy[i]
        #j = list(Testx[i,:]).index(1) # if onehot
        k = int(PredY[i].argmax(axis=0))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(MODULATIONS)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
        sub_Acc.loc[snr][i] = confnorm[i][i]
    #plt.figure(figsize=(680,500))
    plot_confusion_matrix(confnorm, labels=MODULATIONS.keys(), title=f"AMR(SNR=%d)"%(snr), SNR=snr)
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 100*cor/(cor+ncor)
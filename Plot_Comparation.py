import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pathimg = './sava_img'
def plot_bycolumns(plot_data, mark, title):#plot by columns
    #plt.figure()
    for i in range(1, 8):
        plt.plot(plot_data[plot_data.columns[0]], plot_data[plot_data.columns[i]], label=plot_data.columns[i], marker=mark[i])
        # if average is True:
        #     avg = [plot_data[plot_data.columns[i]].mean()]*len(m)
        #     plt.plot(m, avg, linestyle = '--', label=f'{labels[i]}:average', marker=mark[i])
    plt.yticks(np.arange(start=0, stop=100, step=10))
    plt.xticks(np.arange(start=-20, stop=20, step=4), rotation=0)
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig(f'{pathimg}/{title}.png')
    plt.show()
#data_name = [f'Conformer{i}_2_CR={c},norescovacc' for i in [9, 12] for c in [2, 4]]
mark = ['>', 'o', '.', '^', 'v', 's', 'p', '*', 'h', '+', 'D', '<']
data_path = './save_csvdata/Comparison.xlsx'
df_Comp = pd.read_excel(data_path)
plot_bycolumns(df_Comp, mark=mark, title='Comparison with other methods')
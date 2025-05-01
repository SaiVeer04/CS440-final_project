import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')

#accuracy vs % data
plt.figure()
for (fw, model), group in df.groupby(['framework','model']):
    for ds in ['digits','faces']:
        sub = group[group['dataset']==ds]
        plt.errorbar(sub['percent'], sub['acc_mean'], yerr=sub['acc_std'], marker='o', label=f'{fw}-{model}-{ds}')
plt.title('Accuracy vs. % of Training Data')
plt.xlabel('Training Data %')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_curve.png')
plt.show()

#training time vs % data
plt.figure()
for (fw, model), group in df.groupby(['framework','model']):
    for ds in ['digits','faces']:
        sub = group[group['dataset']==ds]
        plt.plot(sub['percent'], sub['time_mean'], marker='o', label=f'{fw}-{model}-{ds}')
plt.title('Training Time vs % of Training Data')
plt.xlabel('Training Data %')
plt.ylabel('Training Time s')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('time_curve.png')
plt.show()
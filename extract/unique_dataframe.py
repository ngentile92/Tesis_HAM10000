import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from settings.parameters import DATA_PATH, dict_lesiones


ham10000_metadata = pd.read_csv(DATA_PATH / 'HAM10000_metadata.csv')
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(DATA_PATH, '*', '*.jpg'))}

ham10000_metadata['path'] = ham10000_metadata['image_id'].map(imageid_path_dict.get)
ham10000_metadata['cell_type'] = ham10000_metadata['dx'].map(dict_lesiones.get)
ham10000_metadata['cell_type_idx'] = pd.Categorical(ham10000_metadata['cell_type']).codes

ham10000_metadata.isna().sum()

ham10000_metadata['age'].fillna((ham10000_metadata['age'].mean()), inplace=True)

sns.set_style('whitegrid')
colors = ['#87ace8','#e3784d', 'green']
fig,axes = plt.subplots(figsize=(12,8))

ax = sns.countplot(x='sex',data=ham10000_metadata, palette = 'Paired')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Gender-wise Distribution')
plt.xticks(rotation=45)
plt.show()

sns.set_style('whitegrid')
fig,axes = plt.subplots(figsize=(12,8))
ax = sns.countplot(x='cell_type',data=ham10000_metadata, order = ham10000_metadata['cell_type'].value_counts().index, palette = 'Paired')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Cell Types Skin Cancer Affected patients')
plt.xticks(rotation=45)
plt.show()

sns.set_style('whitegrid')
fig,axes = plt.subplots(figsize=(12,8))
ax = sns.countplot(x='cell_type',hue='sex', data=ham10000_metadata, order = ham10000_metadata['cell_type'].value_counts().index, palette = 'Paired')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Cell Types Frequencies')
plt.xticks(rotation=45)
plt.show()

sns.set_style('whitegrid')
fig,axes = plt.subplots(figsize=(12,8))
ax = sns.countplot(x='dx_type',data=ham10000_metadata, order = ham10000_metadata['dx_type'].value_counts().index, palette = 'flare')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Cell Types Frequencies')
plt.xticks(rotation=45)
plt.show()
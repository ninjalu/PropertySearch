
# %%
import pandas as pd
import os
# %%
"""
This program creates a csv file of property image paths and ids
"""


def image_list(path, out_name='image_list.csv'):
    files = [(f.path, f.name.split('-')[1])
             for f in os.scandir(path)
             if f.is_file() and f.path.split('/')[-1] != '.DS_Store'
             ]
    df = pd.DataFrame(data=files, columns=['file_path', 'id'])
    df.to_csv(path + out_name, index=False)


if __name__ == '__main__':
    image_list('propertyimages/images/')

# %%

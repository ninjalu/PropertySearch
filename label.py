# %%
import torch
import os
import pandas as pd
# %%


def create_csv(root='./data/', out_name='labels.csv'):
    """Creates a CSV file where each row contains an image file path and it's corresponding integer class label"""
    subfolders = [f.path for f in os.scandir(root) if f.is_dir(
    )]  # get the path of the subfolders in the data root (each of which contains images for certain class)
    # create empty dataframe with file_path and label columns
    df = pd.DataFrame(columns=['file_path', 'label'])
    for i, path in enumerate(subfolders):
        files = [f.path for f in os.scandir(
            path) if f.is_file() and f.path.split('/')[-1] != '.DS_Store']
        for f in files:
            # add each image as a row to the dataframe
            df = df.append({'file_path': f, 'label': i}, ignore_index=True)
    df.to_csv(root+out_name, index=False)  # save the dataframe to a csv file


if __name__ == '__main__':
    create_csv(root='./propertyimages/')
# %%

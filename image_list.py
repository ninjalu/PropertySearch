
import pandas as pd
import os


def image_list(path, out_name='image_list.csv'):
    """creates a csv file where each line is a path of an image"""
    df = pd.DataFrame(columns=['file_path'])
    files = [f.path for f in os.scandir(
        path) if f.is_file() and f.path.split('/')[-1] != '.DS_Store']
    for f in files:
        # add each image as a row to the dataframe
        df = df.append({'file_path': f}, ignore_index=True)
    df.to_csv(path + out_name, index=False)


if __name__ == '__main__':
    image_list('propertyimages/garden')

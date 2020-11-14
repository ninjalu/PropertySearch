# %%
import pandas as pd
import os
import shutil
# %%


def move_image(old_file_path, new_file_path):
    if not os.path.isdir(new_directory):
        os.mkdir(new_directory)
    base_name = os.path.basename(old_file_path)
    new_file_path = os.path.join(new_directory, base_name)
    # Deletes a file if that file already exists there, you can change this behavior
    if os.path.exists(new_file_path):
        os.remove(new_file_path)
    os.rename(old_file_path, new_file_path)


# %%
home_path = '/Users/luluo/projects/propertySearch/propertyimages/'

df = pd.read_csv('class.csv', index_col=0)
df.columns = ['file_path', 'label']
# %%
for row in df.iterrows():
    old_file_path = home_path + 'images/'
    file_name = row[1][0].split('/')[-1]
    new_file_path = home_path + row[1][1]
    # print(old_file_path)
    # print(new_file_path)
    shutil.copy(os.path.join(old_file_path, file_name), new_file_path)
# %%
# %%

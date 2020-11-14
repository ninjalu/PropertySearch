# %%
from fastai.vision.widgets import *
from fastbook import *
from fastai.vision.all import *
import fastbook
fastbook.setup_book()

# %%


class DataLoaders(GetAttr):
    def __init__(self, *loaders):
        self.loaders = loaders

    def __getitem__(self, idx):
        return self.loaders[idx]

    train, valid = add_props(lambda i, self: self[i])


prop = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(256, min_scale=0.7)
)


data_dir = 'propertyimages/fastai'
data_loader = prop.dataloaders(data_dir)
# data_loader.valid.show_batch(max_n=4, nrows=1, unique=True)

# %%
learn = cnn_learner(data_loader, resnet50, metrics=error_rate)
learn.fine_tune(1)

# %%

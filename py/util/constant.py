import os

# Path constant
ASSET_DIR_NAME = 'asset'
OUT_DIR_NAME = 'out'
EXPERIMENT_DIR_NAME = 'experiment'

# Path
EXPERIMENT_PATH = os.path.join(OUT_DIR_NAME, EXPERIMENT_DIR_NAME)

# OfficeHome dataset
OFFICEHOME_DIR_PATH = os.path.join(ASSET_DIR_NAME, 'OfficeHome')
OFFICEHOME_DATA_PATH = os.path.join(OFFICEHOME_DIR_PATH, 'data')
OFFICEHOME_IMAGE_LIST_PATH = os.path.join(OFFICEHOME_DIR_PATH, 'image_list')

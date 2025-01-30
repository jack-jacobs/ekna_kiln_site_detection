

### Setting paths and bools

import os
import sys

### Import libraries
    
syspath = f"{os.environ['HOME']}/ekna_kiln_detect/VerhegghenReplica/scripts/GDAL-python"
sys.path.append(syspath)

import shapefile
from pathlib import Path

# Common libraries
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from osgeo import osr, ogr, gdal

import json
import pandas as pd
# import geopandas as gpd
import random
import glob
import shutil
import cv2
import numpy as np

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Import torch just to check that version & cuda are what we want
import torch

# Import psycopg2 for PostgreSQL manipulation
# import psycopg2
# from psycopg2 import sql
# import subprocess



### Define functions to run the best model for each dataset
# Load model configuration yaml

def load_conf_file(path):
    config_file_path = path
    # The power value applied to image_count when calcualting frequency weight
    weights_path = "{}model_final.pth".format(path.split('config.yaml')[0])
    cfg = get_cfg()
    cfg.set_new_allowed(True)

    cfg.merge_from_file(config_file_path)
    cfg.DATASETS.TRAIN = ("swalim_train", )
    cfg.DATASETS.TEST = ("swalim_test", )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg

# Make inferences    
def Inference(config, path_list_imgs, table, path_copy_im):
    cfg = load_conf_file(config)
    predictor = DefaultPredictor(cfg)

    with open(path_list_imgs) as f:
        lines = f.read().splitlines()

    for d in lines: 
        im = cv2.imread(d)
        outputs = predictor(im)  
        out = outputs["instances"].to("cpu")
        box = out.pred_boxes
        scores = out.scores
        classes = out.pred_classes.tolist() 
        boxes =  box.tensor.detach().numpy()
        print(boxes)
        if len(boxes>0):
            control = np.zeros((im.shape[0],im.shape[1]), dtype=int)
            for i in range(len(boxes)):
                control[int(boxes[i][1]): int(boxes[i][3]), int(boxes[i][0]): int(boxes[i][2]),] = 1

            name =  '{}{}'.format(path_copy_im, d.split('.tif')[0].split('/')[-1])
            shapefile.ArrayToPoly(d,control,name, path_copy_im, config, scores)


# Wrapper which mostly points Inference() to directories 
def Inference_all(t, df, results_path, cohort, withoutann = True, withann = True): 
    # if runned on another platform pick the path and change it,
    # if not directly df['Path']
    config = df['Path'].replace('/mnt/content/drive/MyDrive/JRC/Swalim_project/swalim_final_clean/outputs/second_iter/pancro_300/', results_path)
    
    #Inference on non annotated images
    if withoutann == True:
        if os.getenv('RGB') == 'False':
            orig_img_dir = f"{os.environ['IN_DIR']}/inputs/{cohort}/pancro/img_without_ann"
            path_list_imgs = f"{os.environ['IN_DIR']}/inputs/{cohort}/withoutann/list_tiles_pancro.csv"
            json_data_path = f"{os.environ['OUT_DIR']}/outputs/{cohort}/Inference/pancro_inf/pancro_inf_{t}.json"
            path_copy_im = f"{os.environ['OUT_DIR']}/outputs/{cohort}/Inference/pancro_inf_{t}/"
            table = f"pancro_{t}".format(t)

        else:
            orig_img_dir = f"{os.environ['IN_DIR']}/inputs/{cohort}/RGB/img_without_ann"
            path_list_imgs = f"{os.environ['IN_DIR']}/inputs/{cohort}/withoutann/list_tiles_RGB.csv"
            json_data_path = f"{os.environ['OUT_DIR']}/outputs/{cohort}/Inference/rgb_inf/rgb_inf_{t}.json"
            path_copy_im = f"{os.environ['OUT_DIR']}/outputs/{cohort}/Inference/rgb_inf_{t}/"
            table = f"rgb_{t}".format(t)
            
        # creating a new directory called pythondirectory
        Path(path_copy_im).mkdir(parents=True, exist_ok=True)
        Inference(config, path_list_imgs, table, path_copy_im)    

    # Inference on annotated images
    if withann == True:
        if os.getenv('RGB') == 'False':
            orig_img_dir = f"{os.environ['IN_DIR']}/inputs/{cohort}/pancro/img_without_ann"
            path_list_imgs = f"{os.environ['IN_DIR']}/inputs/{cohort}/withann/list_tiles_pancro.csv"
            json_data_path = f"{os.environ['OUT_DIR']}/outputs/{cohort}/Inference/pancro_inf/pancro_inf_{t}.json"
            path_copy_im = f"{os.environ['OUT_DIR']}/outputs/{cohort}/Inference/pancro_inf_{t}/"
            table = f"pancro_{t}"

        else:
            orig_img_dir = f"{os.environ['IN_DIR']}/inputs/{cohort}/RGB/img_without_ann"
            path_list_imgs = f"{os.environ['IN_DIR']}/inputs/{cohort}/withann/list_tiles_RGB.csv"
            json_data_path = f"{os.environ['OUT_DIR']}/outputs/{cohort}/Inference/rgb_inf/rgb_inf_{t}.json"
            path_copy_im = f"{os.environ['OUT_DIR']}/outputs/{cohort}/Inference/rgb_inf_{t}/"
            table = f"rgb_{t}"
            
        Path(path_copy_im).mkdir(parents=True, exist_ok=True)
        Inference(config, path_list_imgs, table, path_copy_im)
        
def main():
    
    cohort = sys.argv[1] # e.g. p1
    
    ## DIR is project path
    ## RGB is flag for RGB or panchromatic images

    # os.environ["DIR"] = /eos/jeodpp/data/projects/REFOCUS/data/swalim_v2
    os.environ["DIR"] = f"{os.environ['HOME']}/ekna_kiln_detect"
    os.environ["IN_DIR"] = f"{os.environ['HOME']}/ekna_kiln_detect"
    os.environ["OUT_DIR"] = f"/data4/shared/ekna_kiln_drive/test_cohort"
    os.environ["RGB"] = "True"
    
    # Setup detectron logger
    setup_logger()
    
    # Ensure PyTorch is loaded
    print(torch.__version__)
    print(torch.cuda.is_available())

    # Create results path
    # This path contains the csv files from Notebook 3
    if os.getenv('RGB') == 'False':
        results_path = f"{os.getenv('OUT_DIR')}/{cohort}/pancro"
    else:
        results_path = f"{os.getenv('OUT_DIR')}/{cohort}/RGB"
        
    print(results_path)
    
    ### Define inputs to Inference_all() & run
    skip_inf_for_now = False

    if os.getenv('RGB') == 'True':
        config_yml_path = "/data4/shared/ekna_kiln_drive/V-MS_param_weights/RGB/config.yaml"
    else:
        config_yml_path = "/data4/shared/ekna_kiln_drive/V-MS_param_weights/pancro/config.yaml"


    if skip_inf_for_now == False:
        t = "cohort"

        input_df = pd.DataFrame([{
            "Path": config_yml_path,
            # Additional fields expected by Inference_all #
        }])

        Inference_all(t, input_df.iloc[0], results_path, cohort, withann = False)
        
if __name__ == '__main__':    
    main()    
    print('All done!')
    

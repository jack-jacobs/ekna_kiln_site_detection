import os
import pandas as pd
import shutil

# Define paths
parent_dir = f'{os.environ['HOME']}/ekna_kiln_detect/subimage_selection'
files_dir = '/data4/shared/ekna_kiln_drive/images/selected_subimages/'
csv_path = os.path.join(parent_dir, 'total.csv')

# Read CSV
grouping = pd.read_csv(csv_path)

# Ensure the output directories exist and organize files
for group_name in grouping['group'].unique():
    group_dir = os.path.join(files_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)
    
# Move files to corresponding directories based on grouping
for _, row in grouping.iterrows():
    filename = f'{row['library_name']}_{row['tileName']}.tif'
    group_name = row['group']
    src = os.path.join(files_dir, filename)
    dest = os.path.join(files_dir, group_name, filename)
    
    # Ensure the file exists before moving
    if os.path.exists(src):
        shutil.move(src, dest)
    # else:
    #     print(f'File not found: {src}')

print('Files have been organized into subdirectories by group.')

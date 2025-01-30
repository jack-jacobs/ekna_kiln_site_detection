# subimage_selection.py
# Author: Jack Jacobs (jackjaco@sas.upenn.edu)
# 
# Produce CSV of the selection of G-EGD subimages that intersect with
# the Zambia EKNA kiln site detection project's geographic area of interest.
# 
# This script takes 4 arguments from the command line after the script name
# in the following order:
#   1) The path to the lookup table for the images
#   2) The path to the project area of interest
#   3) The directory path containing all the images' subimage shapefiles
#   4) The path to the shapefile for that special zone of interest H shared
#   5) The output path for the resulting CSV

# Utility libraries
import os
import sys

# Data manipulation libraries
import numpy as np
import pandas as pd
import geopandas as gpd

# Define main execution flow
def main(args: tuple[str, str, str, str]) -> None:
    
    # Read in the lookup table
    lookup = pd.read_csv(args[0])
    
    # Read in the project area of interest
    aoi = gpd.read_file(args[1])
    
    # Define the list of tileshape geometries to be concatenated
    gdf_list = [
        gpd.read_file(f'{args[2]}/{path}') \
            for path in os.listdir(args[2])
    ]
    
    # Make the huge gdf by reading in then concatenating all tileshapes
    gdf = (pd.concat(gdf_list)
           .merge(lookup, left_on = 'prodDesc', right_on = 'library_name')
           .drop(columns = ['fileName','prodDesc','volNum'])
           .to_crs(aoi.crs))
    
    # Add T/F value for whether or not each subimage intersects with AOI
    gdf['relevant'] = gdf.geometry.intersects(aoi.geometry.iloc[0])
    
    # From this result, see which geometries intersect with
    #  the economic and conservation zones Heather highlighted
    special = gpd.read_file(args[3])
    gdf['priority'] = gdf.geometry.intersects(
        special.dissolve().geometry.iloc[0]
    )
    
    # For the result, keep only the rows that intersect with the AOI
    result = gdf[gdf.relevant]
    
    # Export the resulting gdf to CSV
    (result
     .drop(columns=['geometry','relevant'])
     .to_csv(args[4], index = False))
    
    # Print numbers involved to console
    print(
        f'\nSubimage selection from {len(gdf_list)} images reduced from '
        f'{gdf.shape[0]} subimages to {result.shape[0]} subimages.\n'
        f'Of these subimages, {result.priority.sum()} are priority.\n'
    )

# Execute script
if __name__ == '__main__':
    main(sys.argv[1:])

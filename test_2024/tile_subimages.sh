#!/bin/bash
#SBATCH -J ekna_tiling
#SBATCH --partition=gpc2_compute
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH -o test_2024/output.log
#SBATCH -e test_2024/error.log
#SBATCH --time=2:00:00

# Run this script from ~/ekna_kiln_detect

# Activate conda environment for tiling
source ../miniconda3/etc/profile.d/conda.sh
conda activate kiln

# Remove BROWSE files if they exist
find . -name "*BROWSE*" -delete

# Programatically generate command strings for all subimages
SIZE=512
JOBS=8
OVERLAP=20
mkdir -p scripts_toerase/$1
mkdir -p inputs/$1/Tiled
chmod +x VerhegghenReplica/scripts/processing/*.sh
for file in /data4/shared/ekna_kiln_drive/images/selected_subimages/$1/*.tif
do
    echo VerhegghenReplica/scripts/processing/maketiles.sh ${file} $SIZE $(basename "$file" .tif) $OVERLAP inputs/$1/Tiled/ > scripts_toerase/$1/'mt_'$(basename "$file")'.sh'
done

# Make sure all these tiling scripts are executable
chmod +x scripts_toerase/$1/mt_*.sh

# Add all script execution calls to a single text file and print the number of lines
ls scripts_toerase/$1/mt_* > scripts_toerase/$1/Main-par.txt
ls scripts_toerase/$1/mt* | wc -l

echo "All tiling scripts prepped."

# Send the tiling commands into a parallel job
cat scripts_toerase/$1/Main-par.txt | "../miniconda3/envs/kiln/bin/parallel" -j $JOBS {}

echo "All tiles created."

# Remove the intermediate script directory and its contents
rm -r scripts_toerase/$1

# List all the tiles
find inputs/$1/Tiled -name "*tif" > inputs/$1/list_tiles.csv
mkdir -p inputs/$1/Tiled/RGB

# Erase all tiles that are 0
python VerhegghenReplica/scripts/processing/erase_tiles_nodata.py $1

# Clean the tiles folders
rm inputs/$1/Tiled/*.xml
rm inputs/$1/Tiled/*.tfw

# Create the footprints of the RGB tiles
rm -r inputs/$1/footprints
mkdir -p inputs/$1/footprints
find inputs/$1/Tiled/RGB -name '*.tif' -print | xargs --max-args=50 gdaltindex -t_srs EPSG:4326 -src_srs_name src_srs inputs/$1/footprints/footprint-RGB.shp
find inputs/$1/Tiled/RGB -type f -maxdepth 1 -name "*.tif" > inputs/$1/list_tiles_RGB.csv

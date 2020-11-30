#
# Downloads all available shape categories and unzips to the current folder.
#

# Cars (11GB compressed, 36GB expanded)
wget http://download.cs.stanford.edu/orion/caspr_data/cars.zip
unzip cars.zip
rm cars.zip
# Chairs (21GB compressed, 70GB expanded)
wget http://download.cs.stanford.edu/orion/caspr_data/chairs.zip
unzip chairs.zip
rm chairs.zip
# Airplanes (15GB compressed, 52GB expanded)
wget http://download.cs.stanford.edu/orion/caspr_data/airplanes.zip
unzip airplanes.zip
rm airplanes.zip
# Warping Cars 8k (22GB compressed, 24GB expanded)
wget http://download.cs.stanford.edu/orion/caspr_data/warping_cars_full_cat_8k.zip
unzip warping_cars_full_cat_8k.zip
rm warping_cars_full_cat_8k.zip
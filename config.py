import os

CONFIG = {
    'num_episodes': 10000,
    'batch_size': 32,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.999,
    'learning_rate': 0.001,
    'target_update_freq': 100,
    'replay_buffer_size': 5000,
    'min_replay_size': 1000,
    'max_steps': 1000
}

# 파일 경로 설정
DEM_FILE = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/dem/dem.tif'
ROAD_FILE = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/road/road_centerline.shp'
FORESTROAD_FILE = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/road/lt_l_frstclimb.shp'
CLIMBPATH_FILE = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/hiking/ulsan_climb_path.shp'
output_path = '/Users/heekim/new_multiagent/new_simulation_path.json'
model_path = '/Users/heekim/new_multiagent/weights/best_model_pointer.pth'
area_difference_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/area/area_difference.shp'

rirsv_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/river/it_c_rirsv.shp'
wkmstrm_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/river/lt_c_wkmstrm.shp'
channels_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/watershed/8/channel8.shp'
watershed_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/watershed/8/validBasins8.shp'

MODEL_PATH = os.path.join('weights', 'best_model_pointer.pth')

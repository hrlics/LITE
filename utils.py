import os
import json
import torch
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.patches as patches


def draw_image(dataset_name, seg_idx, date_idx, ts_values, ts_params, ts_scales, 
                override, differ, outlier, image_size, grid_layout, linestyle, 
                linewidth, markersize, ts_marker_mapping, ts_color_mapping, ts_idx_mapping):
    
    # set matplotlib param
    grid_height = grid_layout[0]
    grid_width = grid_layout[1]
    if image_size is None:
        cell_height = 64
        cell_width = 64
        img_height = grid_height * cell_height
        img_width = grid_width * cell_width
    else:
        img_height = image_size[0]
        img_width = image_size[1]

    dpi = 100
    plt.rcParams['savefig.dpi'] = dpi  
    plt.rcParams['figure.figsize'] = (img_width / dpi, img_height / dpi)
    plt.rcParams['figure.frameon'] = False
    
    base_path = f"dataset/{dataset_name}/images/"

    if not os.path.exists(base_path): os.makedirs(base_path)
    img_path = os.path.join(base_path, f"{seg_idx}_{date_idx}.png")
    if os.path.exists(img_path):
        if not override:
            return img_path

    drawed_params = []

    # find the information across all the pations
    num_params = ts_values.shape[-1]
    ts_orders = list(range(len(ts_params)))
    
    for idx, param_idx in enumerate(ts_orders):
        
        param = ts_params[param_idx]
        ts_value = ts_values[:, param_idx] # (30,)
        
        # the scale of x, y axis
        param_scale_x = [0, ts_value.shape[0]]
        param_scale_y = [np.nanmin(ts_value),np.nanmax(ts_value)]

        if np.isnan(param_scale_y[0]):
            param_scale_y = [-1, 1]
        
        # only one value, expand the y axis
        if param_scale_y[0] == param_scale_y[1]:
            param_scale_y = [param_scale_y[0]-0.5, param_scale_y[0]+0.5]

        ts_time = np.array(list(range(ts_value.shape[0]))).reshape(-1,1)
        ts_value = np.array(ts_value).reshape(-1,1)
        
        ##### draw the plot for each parameter
        param_marker = ts_marker_mapping[param]
        param_color = ts_color_mapping[param]
        param_idx = ts_idx_mapping[param]

        plt.subplot(grid_height, grid_width, idx+1) # 6*6
        if differ: # using different colors and markers
            plt.plot(ts_time, ts_value, linestyle=linestyle, 
                     linewidth=linewidth, markersize=markersize, color=param_color, marker="*")
        else:
            plt.plot(ts_time, ts_value, linestyle=linestyle, linewidth=linewidth)

        # set the scale for x, y axis
        plt.xlim(param_scale_x)
        plt.ylim(param_scale_y)
        plt.xticks([])
        plt.yticks([])

        drawed_params.append(param)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.savefig(img_path, pad_inches=0)
    plt.clf()

    return img_path

def construct_image(
    seg_idx, date_idx, dataset,
    linestyle="-", linewidth=1, markersize=2, 
    override=False, 
    differ=False, 
    outlier=None,
    grid_layout=(3,3),
    image_size=None,
    dataset_name='CRW_Temp'
    ):

    # variables: [ the day of the year, rainfall, daily average air temperature, solar radiation, average cloud cover fraction, ground water temperature, 
    # subsurface temperature, potential evapotranspiration, daily average water temperature]

    ts_params=np.array(['DOY', 'R', 'DAAT', 'SR', 'ACCF', 'GWT', 'ST', 'PE', 'DAWT']) # for CRW-Temp and CRW-Flow
    # ts_params=np.array(['tair','swdown','precip','spRH','n2o']) for AGR
    num_ts_params = len(ts_params)

    # load markers and colors
    f = open(f'dataset/{dataset_name}/plt_markers_desc.json', 'r') # should modify according to your dataset
    plt_markers_description = json.load(f)
    f = open(f'dataset/{dataset_name}/plt_colors_desc.json', 'r')
    plt_colors_description = json.load(f)

    plt_markers = list(plt_markers_description.keys())
    num_markers = len(plt_markers)
    plt_colors = list(plt_colors_description.keys())

    # construct the mapping from param to marker, color, and idx
    ts_marker_mapping = {}
    ts_idx_mapping = {}
    ts_color_mapping = {}
    for idx, param in enumerate(ts_params):
        if idx < num_markers:
            ts_marker_mapping[param] = plt_markers[idx]
        else: # if not enough markers, use (num_sides, 0/1/2, angles) markers
            marker = (int((idx-num_markers)/3)+3, int((idx-num_markers)%3)) # starting from (3,0)
            ts_marker_mapping[param] = marker
        ts_color_mapping[param] = plt_colors[idx]
        ts_idx_mapping[param] = idx

    with open(f'dataset/{dataset_name}/param_marker_mapping.json', 'w') as f:
        json.dump(ts_marker_mapping, f)
    with open(f'dataset/{dataset_name}/param_idx_mapping.json', 'w') as f:
        json.dump(ts_idx_mapping, f)
    with open(f'dataset/{dataset_name}/param_color_mapping.json', 'w') as f:
        json.dump(ts_color_mapping, f)

    all_ts_values = [[] for _ in range(num_ts_params)]
    for param_idx in range(num_ts_params):
        all_ts_values[param_idx] = dataset[:, param_idx] #=dataset[:, param_idx].reshape(dataset[:, param_idx].shape[0]*dataset[:, param_idx].shape[1], -1)

    stat_ts_values = np.ones(shape=(num_ts_params, 10)) # mean, std, y_min, y_max

    for param_idx in range(num_ts_params): 
        param_ts_value = all_ts_values[param_idx]

        stat_ts_values[param_idx,0] = param_ts_value.mean()
        stat_ts_values[param_idx,1] = param_ts_value.std()
        stat_ts_values[param_idx,2] = param_ts_value.min()
        stat_ts_values[param_idx,3] = param_ts_value.max()

        """
        1. remove outliers with boxplot
        """
        q1 = np.percentile(param_ts_value, 25)
        q3 = np.percentile(param_ts_value, 75)
        med = np.median(param_ts_value)
        iqr = q3-q1
        upper_bound = q3+(1.5*iqr)
        lower_bound = q1-(1.5*iqr)
        stat_ts_values[param_idx,4] = lower_bound
        stat_ts_values[param_idx,5] = upper_bound
        param_ts_value1 = param_ts_value[(lower_bound<param_ts_value)&(upper_bound>param_ts_value)]
        outlier_ratio = 1 - (len(param_ts_value1) / len(param_ts_value))

        """
        2. remove outliers with standard deviation
        """
        med = np.median(param_ts_value)
        std = np.std(param_ts_value)
        upper_bound = med + (3*std)
        lower_bound = med - (3*std)
        stat_ts_values[param_idx,6] = lower_bound
        stat_ts_values[param_idx,7] = upper_bound
        param_ts_value2 = param_ts_value[(lower_bound<param_ts_value)&(upper_bound>param_ts_value)]
        outlier_ratio = 1 - (len(param_ts_value2) / len(param_ts_value))

        """
        3. remove outliers with modified z-score
        """
        med = np.median(param_ts_value)
        deviation_from_med = param_ts_value - med
        mad = np.median(np.abs(deviation_from_med))
        lower_bound = (-3.5/0.6745)*mad + med
        upper_bound = (3.5/0.6745)*mad + med
        stat_ts_values[param_idx,8] = lower_bound
        stat_ts_values[param_idx,9] = upper_bound
        param_ts_value3 = param_ts_value[(lower_bound<param_ts_value)&(upper_bound>param_ts_value)]
    
    
    # second round, draw the image for each datapoint
    ts_values = dataset
    
    # normalize the values
    if not outlier:
        ts_scales = stat_ts_values[:,2:4] # no removal
    elif outlier == "iqr":
        ts_scales = stat_ts_values[:,4:6] # iqr
    elif outlier == "sd":
        ts_scales = stat_ts_values[:,6:8] # sd
    elif outlier == "mzs":
        ts_scales = stat_ts_values[:,8:10] # mzs
    
    # draw the image for each p
    image_path = draw_image(dataset_name, seg_idx, date_idx, ts_values, ts_params, ts_scales, 
                                override, differ, outlier,
                                image_size, grid_layout, 
                                linestyle, linewidth, markersize,
                                ts_marker_mapping, ts_color_mapping, ts_idx_mapping)
    
    return image_path


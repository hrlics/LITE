import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import Image, load_from_disk
from utils import construct_image


def Draw(args):

    np.random.seed(args.seed) #3407
    l_past = args.len_look_back

    X = np.load(args.feature_path)  # z: [n_locations, n_days, n_feat]
    Y = np.load(args.target_path) # z: [n_locations, n_days]
    n_locs = X.shape[0]
    n_days = X.shape[1]

    X, Y = X.reshape(-1, X.shape[-1]), Y.flatten()
    X_Y = np.concatenate((X, Y.reshape(-1, 1)), axis=1).reshape(n_locs, n_days, -1) # z: [n_locations, n_days, n_feat+1]
    
    data_for_image=[]
    for loc_idx, loc in enumerate(X_Y):
        for date_idx, date in enumerate(loc):
            
            if date_idx >= l_past:
                past_records = X_Y[loc_idx, date_idx-l_past:date_idx]
                data_for_image.append(past_records)
    data_for_image=np.stack(data_for_image)
    
    print(f"\n== Ready to draw image, data shape: {data_for_image.shape}\n")
    _draw_img(data_for_image, n_days, l_past, args.dataset_name, args.dataset_path)


def _draw_img(data, n_days, l_past, dataset_name, dataset_path):
    img_paths=[]
    dataset = load_from_disk(dataset_path)        
    dataset_folder = f'./dataset/{dataset_name}'
    
    print("dataset_folder", dataset_folder)

    for item in tqdm(dataset):

        if item['date_idx'] < l_past or item['label']==-11. or np.isnan(item['label']): # if item['hour_idx'] < l_past:
            img_paths.append(f"./dataset/{dataset_name}/images/None.png") # all white
        
        elif item['date_idx'] >= l_past:  #elif item['hour_idx'] >= l_past:
            image_data = data[
                item['seg_idx'] * n_days + item['date_idx'] - l_past * (item['seg_idx']+1)
            ]  
            # item['chamber_idx'] * n_days + item['hour_idx'] - l_past * (item['chamber_idx'] + 1) 
            
            image_path = construct_image(
                item['seg_idx'], item['date_idx'], image_data, # item['chamber_idx'], item['hour_idx'], image_data,
                linestyle="-", linewidth=1, markersize=2, 
                override=True,
                differ=True, 
                outlier=None,
                grid_layout=(3,3),  # grid_layout=(2,3),
                image_size=(224, 224),
                dataset_name=dataset_name
            )
            img_paths.append(image_path)
            print(image_path)
            break

    dataset = dataset.add_column("image", img_paths)
    dataset = dataset.cast_column("image", Image())
    dataset.save_to_disk(f'./dataset/{dataset_name}/{dataset_name}_wImage')
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default=None)
    parser.add_argument("--target_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--len_look_back", type=int, default=30, help="look back window size of temporal trend image")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    Draw(args)
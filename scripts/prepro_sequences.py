import pandas as pd
import numpy as np
import argparse
import cv2
import json
import base64
import sys
import csv

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

def get_detection_from_images(real_obj_df, img_df, feats, image_path, this_image):
    '''
    construct input for .tsv files
    '''
    boxes = real_obj_df[real_obj_df.image_id == this_image]['bb'].tolist()

    boxes = [[box[0], box[1], box[3], box[2]] for box in boxes]
    boxes = np.asarray(boxes, dtype=np.float32)
    # 212, 1078

    filename = img_df[img_df.image_id == this_image]['filename'].item()
    split = img_df[img_df.image_id == this_image]['split'].item()
    cat = img_df[img_df.image_id == this_image]['image_cat'].item()
    image = image_path+split+'/'+cat+'/'+filename
    im = cv2.imread(image)
    height, width = im.shape[:2]

    num_boxes = real_obj_df[real_obj_df.image_id == this_image].shape[0]

    features_comb = []
    for item in feats:
        if this_image == int(item[1]):
            obj_features = item[3:-7]
            features_comb.append(obj_features)

    features_comb = np.asarray(features_comb, dtype=np.float32)

    return {
        'image_id': this_image,
        'image_h': height,
        'image_w': width,
        'num_boxes' : num_boxes,
        'boxes': base64.b64encode(boxes),
        'features': base64.b64encode(features_comb)
    }

def build_split(split_file, split_type):
    ids = []
    with open(split_file, 'r') as sf:
        spl = json.load(sf)
        ids = [int(key) for key, value in spl.items() if value == split_type]
    return ids
    #print(len(ids))

def objects_to_keep(obj_features, this_split, obj_df):
    indexes_to_keep = []
    for image_id in this_split:
        for item in obj_features:
            if int(item[1]) == image_id:
                level = int(str(int(item[2]))[-1])
                region_id = int(str(int(item[2]))[:-1])
                obj_subset = obj_df[obj_df.image_id == int(item[1])]
                this_object_index = obj_subset[(obj_subset.region_id == region_id) & (obj_subset.level == level)].index.values[0]
                indexes_to_keep.append(this_object_index)
    real_obj_df = obj_df.iloc[indexes_to_keep,:]
    return real_obj_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_df',
                        default='/home/nikolai/049_matchit/content_selection/data_processed/ade_imgdf.json.gz',
                        help='path to the ade images dataframe')
    parser.add_argument('--obj_df',
                        default='/home/nikolai/049_matchit/content_selection/data_processed/ade_objdf.json.gz',
                        help='path to the ade objects dataframe')
    parser.add_argument('--feats',
                        default='/home/nikolai/049_matchit/content_selection/data_processed/ade_obj_regfeats_vgg19-fc2.npz',
                        help='path to the VGG features')
    parser.add_argument('--image_path',
                        default='/media/dsgdata/Corpora/External/ImageCorpora/ADE20K_2016_07_26/images/',
                        help='basepath to the actual images')
    parser.add_argument('--splits',
                        default='/home/nikolai/049_matchit/data/processed/splits.json',
                        help='location of the json with splits')
    parser.add_argument('--split_type',
                        default='train',
                        help='which set to construct .tsv for: train | val | test')

    args = parser.parse_args()

    ade_img = pd.read_json(args.img_df, orient='split', compression='gzip')
    ade_obj = pd.read_json(args.obj_df, orient='split', compression='gzip') 
    image_features = np.load(args.feats)['arr_0']
    


    splits = build_split(args.splits, args.split_type)
    res_obj_dataframe = objects_to_keep(image_features, splits, ade_obj)
    this_file_split = str(args.split_type)

    print('Working on the', args.split_type, 'set...')

    # generate tsv
    with open("./{}.tsv".format(this_file_split), 'ab') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
        for img in splits:
            # check image 1948
            if img != 1948:
                #print(img)
                writer.writerow(get_detection_from_images(res_obj_dataframe, ade_img, image_features, args.image_path, img))
    print('Done with', args.split_type, 'set!')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='/home/nikolai/049_matchit/content_selection', help='downloaded feature directory')
parser.add_argument('--output_dir', default='../data/data/adebu', help='output feature files')

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infiles = ['train.tsv', 'val.tsv', 'test.tsv']

#os.makedirs(args.output_dir+'_att')
#os.makedirs(args.output_dir+'_fc')
#os.makedirs(args.output_dir+'_box')

#test_ids = [10592, 8394, 10383, 10400, 11215]
#train_ids = [815, 10228, 101713, 10471, 10452, 10669, 2861, 6965, 2844, 5709, 5706, 14970, 5767, 10748, 101481, 101522, 11378, 8467, 5711, 10484, 101963, 10644]
#val_ids = [919, 2570, 922]

#exceptions = train_ids + val_ids + test_ids
exceptions = [101987, 101969, 101967, 101966, 101970, 101971, 101968, 101965]

for infile in infiles:
    print('Reading ' + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['num_boxes'] = int(item['num_boxes'])
            if item['image_id'] not in exceptions:
                for field in ['features', 'boxes']:
                    item[field] = np.frombuffer(base64.decodestring(item[field]), 
                        dtype=np.float32).reshape((item['num_boxes'],-1))

                np.savez_compressed(os.path.join(args.output_dir+'_att', str(item['image_id'])), feat=item['features'])
                np.save(os.path.join(args.output_dir+'_fc', str(item['image_id'])), item['features'].mean(0))
                np.save(os.path.join(args.output_dir+'_box', str(item['image_id'])), item['boxes'])



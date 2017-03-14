import sys
from collections import OrderedDict, namedtuple;
import os;
import yaml
import keras
import numpy as np;
import pdb 

#import theano
import deeplift
from deeplift.conversion import keras_conversion as kc 
from deeplift import models
from deeplift.blobs import NonlinearMxtsMode,DenseMxtsMode
from deeplift.util import * 
from deeplift import backend as B 

from itertools import izip
import os
import subprocess
from pysam import tabix_index
import h5py
os.environ['GDL_NOCAFFE'] = ''
sys.path.insert(0,"/users/avanti/caffe/python/")
from genomedatalayer.gdlfile import GDLFile

import pdb
import pyximport
import argparse

_importers = pyximport.install()
from qcatIO import write_score_for_single_interval
pyximport.uninstall(*_importers)

np.random.seed(1234)
#####################################################
# Magic numbers and constants here (that are not in args)
BLOB_DIMS = 4
SEQUENCE_CONV_LAYER_NAME = 'sequence-conv1'

def parse_args():
    parser = argparse.ArgumentParser(description='Make DeepLIFT tracks')

    parser.add_argument('--model_yaml', help='Keras model yaml')
    parser.add_argument('--model_weights', help='Keras model weights (HDF5)')
    parser.add_argument('--model_hdf5',help='Keras model hdf5',default=None) 
    parser.add_argument('--hdf5', help='hdf5 of the input data')
    parser.add_argument('--gdl_file', help='GDL file corresponding to input')
    parser.add_argument('--chromputer_mark',help='Mark name (for chromputer models)',required=True)
    parser.add_argument('--output_dir', help='Output directory for tracks',default='')
    parser.add_argument('--verbose', help='Verbose mode', action='store_true')
    parser.add_argument('--batch_size', type=int,help='batch size')
    parser.add_argument('--default_input_mode_name',help='default_input_mode')
    parser.add_argument('--pre_activation_target_layer_name',help='pre-activation target layer name')
    parser.add_argument('--num_tasks',type=int)
    parser.add_argument('--model_type',help="one of graph,sequential,functional")
    parser.add_argument('--task_subset',help="comma-delimited list of neurons")
    return parser.parse_args()

def reconstitute_model(args):
    if args.model_hdf5!=None:
        import keras 
        model=keras.models.load_model(args.model_hdf5)
    else:
        from keras.legacy.models import * 
        yaml_string=open(args.model_yaml,'r').read()
        model_config=yaml.load(yaml_string)
        model=Graph.from_config(model_config)
        model.load_weights(args.model_weights)
    print('got model!')
    return model

def names_from_mark(model, mark):
    softmax_layer, = [layer for layer in model.nodes.keys()
                      if layer.endswith('{}_softmax'.format(mark))]
    ip_layer = softmax_layer.split('_')[0]
    return mark, mark + '_loss', ip_layer


def expand_dims_blob(arr, target_num_axes=BLOB_DIMS):
    # Reshapes arr, adds dims after the first axis
    assert len(arr.shape) <= BLOB_DIMS
    extra_dims = target_num_axes - len(arr.shape)
    new_shape = (arr.shape[0],) + (1,)*extra_dims + tuple(arr.shape[1:])
    return arr.reshape(new_shape)


def _write_2D_deeplift_track(scores, intervals, file_prefix,first,last,line_id, reorder,
                             categories):
    # Writes out track as a quantitative category series:
    # http://wiki.wubrowse.org/QuantitativeCategorySeries
    # TODO: implement reorder = False
    if not reorder:
        raise NotImplementedError

    assert scores.ndim == 3


    if categories is None:
        categories = np.arange(scores.shape[1])
    #if this is the first time the function is called, open the output file, otherwise append to existing file! 
    if first: 
        with open(file_prefix, 'w') as fp:
            line_id = 0
            for interval, score in izip(intervals, scores):
                line_id = write_score_for_single_interval(fp, interval, score,line_id, categories)
    else: 
        with open(file_prefix, 'a') as fp:
            for interval, score in izip(intervals, scores):
                line_id = write_score_for_single_interval(fp, interval, score,
                                                      line_id, categories)
    #collapse duplicates from the hammock file -- keep the entry with the highest abs(deepLIFT) 
    if last: 
        #bedtools sort the hammock file ! 
        try:
            #sort_command="bedtools sort -i "+file_prefix+".collapsed" 
            sort_command="bedtools sort -i "+file_prefix 
            sort_command=sort_command.split(' ') 
            print(sort_command) 
            with open(file_prefix+'.sorted','w') as outfile: 
                subprocess.call(sort_command,stdout=outfile)
        except subprocess.CalledProcessError as e:
            pass

        compressed_file = tabix_index(file_prefix+".sorted", preset='bed', force=True)
        assert compressed_file == file_prefix+".sorted" + '.gz'
    return line_id 

CHROM_SIZES = '/mnt/data/annotations/by_release/hg19.GRCh37/hg19.chrom.sizes'

def _write_1D_deeplift_track(scores, intervals, file_prefix,first):
    assert scores.ndim == 2

    bedgraph = file_prefix + '.bedGraph'
    if first: 
        with open(bedgraph, 'w',100000) as fp:
            for interval, score in izip(intervals, scores):
                chrom = interval.chrom
                start = interval.start
                #outputline="" 
                for score_idx, val in enumerate(score):
                    #output_line=output_line+chrom+'\t'+str(start+score_idx)+'\t'+str(start+score_idx+1)+'\t'+str(val)+'\n'
                    #'''
                    if abs(val) > 1e-3: 
                        #outputline=outputline+chrom+'\t'+str(start+score_idx)+'\t'+str(start+score_idx+1)+'\t'+str(val)+'\n' 
                        #'''
                        fp.write('%s\t%d\t%d\t%g\n' % (chrom,
                                                       start + score_idx,
                                                       start + score_idx + 1,
                                                       val))
                        #'''
                #fp.write(outputline) 
    else: 
        with open(bedgraph, 'a',100000) as fp:
            #output_line="" 
            for interval, score in izip(intervals, scores):
                chrom = interval.chrom
                start = interval.start
                #outputline=""
                for score_idx, val in enumerate(score):
                    #output_line=output_line+chrom+'\t'+str(start+score_idx)+'\t'+str(start+score_idx+1)+'\t'+str(val)+'\n'
                    #'''
                    if abs(val) > 1e-3: 
                        #outputline=outputline+chrom+'\t'+str(start+score_idx)+'\t'+str(start+score_idx+1)+'\t'+str(val)+'\n'
                        fp.write('%s\t%d\t%d\t%g\n' % (chrom,
                                                       start + score_idx,
                                                       start + score_idx + 1,
                                                       val))
                        
                #fp.write(outputline)                 


def write_deeplift_track(scores, intervals, file_prefix, first,last,line_id,reorder=True,
                         categories=None,):
    #if len(scores.shape) != BLOB_DIMS:
    #    pdb.set_trace() 
    #    raise ValueError('scores should have same number of dims as a blob')
    if scores.shape[0] != len(intervals):
        raise ValueError('intervals list should have the same number of '
                         'elements as the number of rows in scores')

    # don't squeeze out the first (samples) dimension
    squeezable_dims = tuple(dim for dim, size in enumerate(scores.shape)
                            if size == 1 and dim > 0)
    scores = scores.squeeze(axis=squeezable_dims)
    signal_dims = scores.ndim - 1
    line_id=0 
    if signal_dims == 2:
       line_id= _write_2D_deeplift_track(scores, intervals, file_prefix,first,last,line_id,reorder=True,categories=categories)
    elif signal_dims == 1:
        _write_1D_deeplift_track(scores, intervals, file_prefix,first)
    else:
        raise ValueError('Cannot handle scores with {} signal dims;'
                         'Only 1D/2D signals supported'.format(signal_dims))
    return line_id 




def partition_intervals(intervals):
    # Partition interval list into several lists of non-overlapping intervals
    # We need this because of subpeaks whose windows will overlap
    # partitions will contain a list of partitions, which is itself a list
    # containing (index, interval) pairs, where the index is such that
    # intervals[index] = interval for the original intervals list passed in.
    from pybedtools import Interval

    def overlap(i1, i2):
        if i1.chrom != i2.chrom:
            return False
        return (i2.start < i1.end and i2.end > i1.start)

    partitions = []
    # sort intervals to get a list of (index, interval) pairs such that
    # intervals[index] = interval
    remaining = sorted(
        enumerate(intervals),
        key=lambda item: (item[1].chrom, item[1].start, item[1].stop))

    while remaining:
        nonoverlapping = [(-1, Interval('sentinel', 0, 0))]
        overlapping = []
        for idx, interval in remaining:
            if not overlap(nonoverlapping[-1][1], interval):
                nonoverlapping.append((idx, interval))
            else:
                overlapping.append((idx, interval))
        partitions.append(nonoverlapping[1:])
        remaining = overlapping
    return partitions

# *** MAIN SCRIPT ***
args = parse_args()
output_name = args.chromputer_mark
output_prefix = os.path.join(args.output_dir, output_name)
try:
    os.mkdir(args.output_dir)
except: 
    print(" output directory exists!")
model=reconstitute_model(args)


# load data with hdf5 generator
inputFile=h5py.File(args.hdf5,'r')  
numentries=inputFile['X'].values()[0].shape[0]
###################################################################################
# load GDL
gdl_file = GDLFile(args.gdl_file)


if args.model_type=="graph": 
    deeplift_model = kc.convert_graph_model(model, nonlinear_mxts_mode=deeplift.blobs.NonlinearMxtsMode.DeepLIFT)
    print(deeplift_model.get_name_to_blob().keys())
    contribs_func=deeplift_model.get_target_contribs_func(find_scores_layer_name=args.default_input_mode_name,pre_activation_target_layer_name=args.pre_activation_target_layer_name)
else:
    deeplift_model=kc.convert_sequential_model(model,nonlinear_mxts_mode=deeplift.blobs.NonlinearMxtsMode.DeepLIFT)
    contribs_func=deeplift_model.get_target_contribs_func(find_scores_layer_idx=0)
    
num_generated=0
input_modes=inputFile['X'].keys()
first=True 
last=False 
line_id_dict=dict() 

for i in range(args.num_tasks): 
    line_id_dict[i]=0

if args.task_subset!=None:
    task_subset=[int(i) for i in args.task_subset.split(',')]
else:
    task_subset=range(args.num_tasks)
    
while num_generated < numentries:
    print("num_generated:"+str(num_generated))
    start_index=num_generated
    end_index=min([numentries,start_index+args.batch_size])
    cur_data_x=[]
    for input_mode in input_modes:
        cur_data_x.append(inputFile['X'][input_mode][start_index:end_index])
    if end_index==(numentries): 
        last=True 
    deeplift_scores = OrderedDict()
    intervals=gdl_file.intervals[start_index:end_index] 
    for neuronOfInterest_idx in task_subset: 
        print("neuronOfInterest_idx:"+str(neuronOfInterest_idx))
        deeplift_scores[neuronOfInterest_idx]=contribs_func(task_idx=neuronOfInterest_idx,input_data_list=cur_data_x,batch_size=args.batch_size,progress_update=100)
    score_type="DeepLIFT"
    for track_name, scores in deeplift_scores.iteritems():
        partition_idx=0 
        file_prefix = '{}_{}_{}_{}'.format(output_prefix, track_name,
                                           score_type, partition_idx)
        scores=np.asarray(scores) #hammock
        #pdb.set_trace() 
        #scores=np.sum(scores,axis=2,keepdims=True) #bigwig 
        line_id_dict[track_name]=write_deeplift_track(scores, intervals, file_prefix,first,last,line_id_dict[track_name])
    first=False 
    num_generated+=(end_index-start_index)
    

import argparse
import json
import yaml 
import h5py 
import keras
import subprocess

def parse_args():
    parser=argparse.ArgumentParser(description='Provide a model yaml & weights files & a dataset, get model predictions and accuracy metrics')
    parser.add_argument('--yaml',help='yaml file that stores model architecture',default=None)
    parser.add_argument('--json',help='json file that stores model architecture',default=None)
    parser.add_argument('--weights',help='hdf5 file that stores model weights',default=None)
    parser.add_argument('--model_hdf5',help="hdf5 file that stores model architecture & weights",default=None)
    parser.add_argument('--model_type',help="graph,functional,sequential")
    parser.add_argument('--dot_file',help="dot file to write model for viewing") 
    return parser.parse_args()

def main():
    args=parse_args()
    #get the model
    if (args.model_hdf5!=None):
        model_hdf5=h5py.File(args.model_hdf5)
        model_config=model_hdf5.attrs.get('model_config')
        model_config=json.loads(model_config.decode('utf-8'))
        model_weights=model_hdf5['model_weights']

        if args.model_type=="graph":
            #from keras.legacy.models import *
            from keras.models import * 
            model=Graph.from_config(model_config)
        else:
            from keras.models import *
            model=model_from_config(model_config)
        model.load_weights_from_hdf5_group(model_weights)

    else:
        if args.yaml!=None:
            yaml_string=open(args.yaml,'r').read()
            model_config=yaml.load(yaml_string)
        else:
            model_config=json.loads(open(args.json).read())
        if args.model_type=="graph":
            from keras.legacy.models import *
            #from keras.models import *
            model=Graph.from_config(model_config)
        else:
            from keras.models import * 
            model=model_from_config(model_config)
        print("got model architecture")
        #load the model weights
        model.load_weights(args.weights)
        print("loaded model weights")
        #plot the model!
    from keras.utils.visualize_util import model_to_dot
    dot_object=model_to_dot(model,show_shapes=True,show_layer_names=True)
    dot_object.write(args.dot_file)
    #use subprocess to plot a png of the dot file
    subprocess.call(["dot","-Tpng","-o",args.dot_file+".png",args.dot_file])
    outf=open(args.dot_file+'.json','w')
    json.dump(model_config,outf)
    

if __name__=="__main__":
    main()
    

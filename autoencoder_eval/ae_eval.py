from keras import backend as K
from keras.models import load_model
import h5py
import numpy as np
import argparse
import pdb

def parse_args():
    parser=argparse.ArgumentParser(description="evaluate auto-encoder")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--num_samples_to_eval",type=int,default=100000)
    parser.add_argument("--layer_to_examine",type=int)
    return parser.parse_args()

def main():
    args=parse_args()

    #real data
    data=h5py.File(args.dataset)
    #just look at first 100k examples
    X=data['X']['sequence'][0:args.num_samples_to_eval]
    #scrambled data
    X_scrambled=np.random.shuffle(X)
    print("got real & scrambled data")

    #load the model
    model=load_model(args.model)
    print("loaded the model") 


    #The layer number
    n = args.layer_to_examine
    # get the layer outputs 
    get_nth_layer_output = K.function([model.layers[0].input],
                                      [model.layers[n].output])
    layer_output = get_nth_layer_output([X])[0]
    pdb.set_trace() 
    
if __name__=="__main__":
    main()
    


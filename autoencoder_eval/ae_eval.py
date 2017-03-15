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
    parser.add_argument('--sample',type=int,help="index of input sequence to examine, 0 by default",default=0)
    parser.add_argument("--layer_to_examine",type=int,help="this defaults to the last layer in the autoencoder, if you want to examine a different layer, provide it here",default=None)
    parser.add_argument("--out_prefix")
    return parser.parse_args()

def main():
    args=parse_args()

    #real data
    data=h5py.File(args.dataset)
    #just look at first 100k examples
    X=np.expand_dims(data['X']['sequence'][args.sample],axis=0)
    #scrambled data
    X_scrambled=np.empty_like(X)
    X_scrambled[:]=X
    
    np.random.shuffle(X_scrambled)
    print("got real & scrambled data")
    print(X.shape)
    print(X_scrambled.shape) 
    #print(X_scrambled) 
    #load the model
    model=load_model(args.model)
    print("loaded the model") 


    #The layer number
    if args.layer_to_examine!=None:
        n=args.layer_to_examine
    else:
        n=len(model.layers)-1
    get_nth_layer_output = K.function([model.layers[0].input],
                                      [model.layers[n].output])
    layer_output = get_nth_layer_output([X])[0]
    scrambled_layer_output=get_nth_layer_output([X_scrambled])[0]
    #save the input & output matrices
    np.savetxt(args.out_prefix+".input.tsv",np.squeeze(X),fmt='%.5f')
    np.savetxt(args.out_prefix+".scrambled.input.tsv",np.squeeze(X_scrambled),fmt='%.5f')
    np.savetxt(args.out_prefix+".output.tsv",np.squeeze(layer_output),fmt='%.5f')
    np.savetxt(args.out_prefix+".scrambled.output.tsv",np.squeeze(scrambled_layer_output),fmt='%.5f')
    
if __name__=="__main__":
    main()
    


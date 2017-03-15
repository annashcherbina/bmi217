from keras import backend as K
from keras.models import load_model
import h5py
import numpy as np
import argparse


def parse_args():
    parser=argparse.ArgumentParser(description="evaluate auto-encoder")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument('--sample',type=int,help="index of input sequence to examine, 0 by default",default=0)
    parser.add_argument("--layer_to_examine",type=int,help="this defaults to the last layer in the autoencoder, if you want to examine a different layer, provide it here",default=None)
    parser.add_argument("--out_prefix")
    parser.add_argument("--plot",action='store_true',default=False)
    return parser.parse_args()

def main():
    args=parse_args()

    #real data
    data=h5py.File(args.dataset)
    X=np.expand_dims(data['X']['sequence'][args.sample],axis=0)
    #scrambled data
    X_scrambled=np.empty_like(X)
    X_scrambled[:]=X
    X_scrambled=np.squeeze(X_scrambled)
    np.random.shuffle(X_scrambled)
    X_scrambled=np.expand_dims(X_scrambled,axis=0)
    X_scrambled=np.expand_dims(X_scrambled,axis=0)
    
    model=load_model(args.model)
  

    #The layer number
    if args.layer_to_examine!=None:
        n=args.layer_to_examine
    else:
        n=len(model.layers)-1
    get_nth_layer_output = K.function([model.layers[0].input],
                                      [model.layers[n].output])
    layer_output = get_nth_layer_output([X])[0]
    scrambled_layer_output=get_nth_layer_output([X_scrambled])[0]
    batch_accuracy=np.sum(X==layer_output)/(1.0*X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3])
    print(str(batch_accuracy))
    #save the input & output matrices
    if args.out_prefix!=None:
        np.savetxt(args.out_prefix+".input.tsv",np.squeeze(X),fmt='%.5f')
        np.savetxt(args.out_prefix+".scrambled.input.tsv",np.squeeze(X_scrambled),fmt='%.5f')
        np.savetxt(args.out_prefix+".output.tsv",np.squeeze(layer_output),fmt='%.5f')
        np.savetxt(args.out_prefix+".scrambled.output.tsv",np.squeeze(scrambled_layer_output),fmt='%.5f')
    #if the user used the --plot flag, generate plots
    if args.plot==True:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        fig=plt.figure()
        plt.imshow(np.squeeze(X)[:,0:20],cmap="hot",interpolation="nearest")
        fig.savefig(args.out_prefix+".input.png")
        fig=plt.figure()
        plt.imshow(np.squeeze(X_scrambled)[:,0:20],cmap="hot",interpolation="nearest")
        fig.savefig(args.out_prefix+".scrambled.input.png")
        fig=plt.figure()
        plt.imshow(np.squeeze(layer_output)[:,0:20],cmap="hot",interpolation="nearest")
        fig.savefig(args.out_prefix+".output.png")
        fig=plt.figure()
        plt.imshow(np.squeeze(scrambled_layer_output)[:,0:20],cmap="hot",interpolation="nearest")
        fig.savefig(args.out_prefix+".scrambled.output.png")
        
if __name__=="__main__":
    main()
    


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
    return parser.parse_args()

def main():
    args=parse_args()

    #real data
    data=h5py.File(args.dataset)
    batch_size=1000
    accuracy=[]
    
    model=load_model(args.model)
    print("loaded the model") 
    
    
    #The layer number
    n=len(model.layers)-1
    get_nth_layer_output = K.function([model.layers[0].input],
                                      [model.layers[n].output])
    for i in range(94): 
        X=np.asarray(data['X']['sequence'][i*batch_size:(i+1)*batch_size])
        print(X.shape)
        layer_output = np.round(get_nth_layer_output([X])[0])
        #check accuracy
        batch_accuracy=np.sum(X==layer_output)/(1.0*X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3])
        accuracy.append(batch_accuracy)
        print(str(batch_accuracy))
    print(str(accuracy))
    print(str(sum(accuracy)/len(accuracy)))

if __name__=="__main__":
    main()
    


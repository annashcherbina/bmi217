#This is a model implemented in "pure keras" for DMSO sequence x matrix x positional values, goal is to see if recallAtFDR50 reproduces results from momma_dragonn 
import imp
import argparse
import numpy as np 
import h5py 
global modelModule
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_path")
    parser.add_argument("--valid_path")
    parser.add_argument("--test_path")
    parser.add_argument("--model_output_file")
    parser.add_argument("--model_builder")
    parser.add_argument("--w0_file")
    parser.add_argument("--w1_file")
    parser.add_argument("--batch_size",type=int,default=1000)
    return parser.parse_args() 

def load_data(args):
    trainmat=h5py.File(args.train_path,'r')
    num_train=args.batch_size*(trainmat['Y']['output'].shape[0]/args.batch_size)
    validmat=h5py.File(args.valid_path,'r')
    num_valid=args.batch_size*(validmat['Y']['output'].shape[0]/args.batch_size) 
    return trainmat,validmat,num_train,num_valid 

        
def fit_and_evaluate(model,train_gen,valid_gen,num_train,num_valid,args):
    model_output_path = args.model_output_file
    checkpointer = ModelCheckpoint(filepath=model_output_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    tensorboard = TensorBoard(histogram_freq=0, write_graph=False, write_images=True)
    csvlogger = CSVLogger('training_results', append = True)
    np.random.seed(1234)
    model.fit_generator(train_gen,
                        validation_data=valid_gen,
                        samples_per_epoch=num_train,
                        nb_val_samples=num_valid,
                        nb_epoch=20,
                        verbose=1,
                        callbacks=[checkpointer,earlystopper, csvlogger, tensorboard])
    print("complete!!") 
    #results = model.evaluate(x_valid,
    #                         y_valid,
    #                         batch_size=1000,
    #                         verbose=1)
    #print str(results)


def main():
    args=parse_args()
    train_mat,valid_mat,num_samples_train,num_samples_valid= load_data(args)
    modelModule=imp.load_source('name',args.model_builder)
    train_generator=modelModule.data_generator(train_mat,args)
    valid_generator=modelModule.data_generator(valid_mat,args) 
    model=modelModule.create_model(args.w0_file,args.w1_file)
    fit_and_evaluate(model,train_generator,
                     valid_generator,
                     num_samples_train,
                     num_samples_valid,args) 

if __name__=="__main__":
    main() 

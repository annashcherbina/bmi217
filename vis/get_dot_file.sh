#AUTOENCODER (David) 
#CUDA_VISIBLE_DEVICES=7 python get_dot_file.py --model_hdf5 optimal_CAE_model.hdf5 --model_type functional --dot_file optimal_CAE_model.dot

#Basset (David)
#CUDA_VISIBLE_DEVICES=7 python get_dot_file.py --model_hdf5 optimal_deep_learning_model5.hdf5 --model_type functional --dot_file optimal_deep_learning_model5.dot

#DeepBind (Andrew)
#CUDA_VISIBLE_DEVICES=7 python get_dot_file.py --model_hdf5 best_model_deepbind.hdf5 --model_type functional --dot_file best_model_deepbind.dot

#Mamie Model 1
CUDA_VISIBLE_DEVICES=7 python get_dot_file.py --model_hdf5 best_model_model1.hdf5 --model_type functional --dot_file best_model_model1.dot

#Mamie Model 2
CUDA_VISIBLE_DEVICES=7 python get_dot_file.py --model_hdf5 best_model_model2.hdf5 --model_type functional --dot_file best_model_model2.dot

#SxMxpos Anna
CUDA_VISIBLE_DEVICES=7 python get_dot_file.py --yaml record_20_model_zGDVe_modelYaml.yaml --weights record_20_model_zGDVe_modelWeights.h5 --model_type graph --dot_file record20.dot

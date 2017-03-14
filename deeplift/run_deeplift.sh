#THEANO_FLAGS="device=gpu7" python run_deeplift.py --model_hdf5 ../models/optimal_deep_learning_model5.hdf5 --hdf5 ref.seq.inputs.hdf5 --gdl_file ref.seq.inputs.gdl --output_dir david.bassetlike  --batch_size 100 --num_tasks=61 --chromputer_mark david.bassetlike --model_type sequential --task_subset=13,15,20,37,44,46


#THEANO_FLAGS="device=gpu3" python run_deeplift.py --model_hdf5 ../models/optimal_deep_learning_model5.hdf5 --hdf5 var.seq.inputs.hdf5 --gdl_file ref.seq.inputs.gdl --output_dir david.bassetlike.vars  --batch_size 100 --num_tasks=61 --chromputer_mark david.bassetlike.vars --model_type sequential --task_subset=13,15,20,37,44,46


#THEANO_FLAGS="device=gpu7" python run_deeplift.py --model_yaml ../models/record_20_model_zGDVe_modelYaml.yaml --model_weights ../models/record_20_model_zGDVe_modelWeights.h5 --hdf5 normalized.gecco.deeplift.vars.hdf5 --gdl_file ref.seq.inputs.gdl --output_dir anna.positional  --batch_size 100 --num_tasks=61 --chromputer_mark anna.positional --model_type graph --task_subset=13,15,20,37,44,46 --default_input_mode_name sequence --pre_activation_target_layer_name fc_5

#THEANO_FLAGS="device=gpu7" python run_deeplift.py --model_yaml ../models/record_20_model_zGDVe_modelYaml.yaml --model_weights ../models/record_20_model_zGDVe_modelWeights.h5 --hdf5 normalized.gecco.deeplift.pos.var.hdf5 --gdl_file ref.seq.inputs.gdl --output_dir anna.positional.vars  --batch_size 100 --num_tasks=61 --chromputer_mark anna.positional --model_type graph --task_subset=13,15,20,37,44,46 --default_input_mode_name sequence --pre_activation_target_layer_name fc_5


######FULL#######################
#THEANO_FLAGS="device=gpu7" python run_deeplift.py --model_hdf5 ../models/optimal_deep_learning_model5.hdf5 --hdf5 ref.seq.inputs.hdf5 --gdl_file ref.seq.inputs.gdl --output_dir david.bassetlike.full  --batch_size 100 --num_tasks=61 --chromputer_mark david.bassetlike --model_type sequential 


#THEANO_FLAGS="device=gpu3" python run_deeplift.py --model_hdf5 ../models/optimal_deep_learning_model5.hdf5 --hdf5 var.seq.inputs.hdf5 --gdl_file ref.seq.inputs.gdl --output_dir david.bassetlike.vars.full  --batch_size 100 --num_tasks=61 --chromputer_mark david.bassetlike.vars --model_type sequential 


#THEANO_FLAGS="device=gpu0" python run_deeplift.py --model_yaml ../models/record_20_model_zGDVe_modelYaml.yaml --model_weights ../models/record_20_model_zGDVe_modelWeights.h5 --hdf5 normalized.gecco.deeplift.pos.hdf5 --gdl_file ref.seq.inputs.gdl --output_dir anna.positional.full  --batch_size 100 --num_tasks=61 --chromputer_mark anna.positional --model_type graph  --default_input_mode_name sequence --pre_activation_target_layer_name fc_5

THEANO_FLAGS="device=gpu1" python run_deeplift.py --model_yaml ../models/record_20_model_zGDVe_modelYaml.yaml --model_weights ../models/record_20_model_zGDVe_modelWeights.h5 --hdf5 normalized.gecco.deeplift.pos.var.hdf5 --gdl_file ref.seq.inputs.gdl --output_dir anna.positional.vars  --batch_size 100 --num_tasks=61 --chromputer_mark anna.positional.full --model_type graph  --default_input_mode_name sequence --pre_activation_target_layer_name fc_5

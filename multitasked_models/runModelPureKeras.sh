#new keras, functional API
#CUDA_VISIBLE_DEVICES=3 python runModelPureKeras.py --train_path inputs/s_by_m_by_pos/dmso.train.hdf5 --valid_path inputs/s_by_m_by_pos/dmso.validate.hdf5 --model_output_file purekeras.model2.hdf5 --model_builder dmso_model.py --w0_file w0.txt --w1_file w1.txt --batch_size 1000

#new keras, legacy Graph API
#CUDA_VISIBLE_DEVICES=7 python runModelPureKeras.py --train_path inputs/s_by_m_by_pos/dmso.train.hdf5 --valid_path inputs/s_by_m_by_pos/dmso.validate.hdf5 --model_output_file purekeras.legacy.graph.hdf5 --model_builder dmso_legacy_graph_model.py --w0_file w0.txt --w1_file w1.txt --batch_size 1000

#functional keras model w/ regularization
#CUDA_VISIBLE_DEVICES=0 python runModelPureKeras.py --train_path inputs/s_by_m_by_pos/dmso.train.hdf5 --valid_path inputs/s_by_m_by_pos/dmso.validate.hdf5 --model_output_file purekeras.functinoal.regularized.hdf5 --model_builder dmso_model_regularized.py --w0_file w0.txt --w1_file w1.txt --batch_size 1000


#functional keras model w/ 1e-6 regularization & normalized data
CUDA_VISIBLE_DEVICES=3 python runModelPureKeras.py --train_path inputs/s_by_m_by_pos/normalized.dmso.train.hdf5 --valid_path inputs/s_by_m_by_pos/normalized.dmso.validate.hdf5 --model_output_file purekeras.NEW.functional.1e-6.hdf5 --model_builder dmso_model_regularized.py --w0_file w0.txt --w1_file w1.txt --batch_size 1000


#CUDA_VISIBLE_DEVICES=3 python runModelPureKerasOld.py --train_path inputs/s_by_m_by_pos/dmso.train.hdf5 --valid_path inputs/s_by_m_by_pos/dmso.validate.hdf5 --model_output_file purekeras.OLD.regularized.1e-6.hdf5 --model_builder dmso_legacy_graph_modelOld.py --w0_file w0.txt --w1_file w1.txt --batch_size 1000

#LEGACY GRAPH MODEL
#CUDA_VISIBLE_DEVICES=7 python runModelPureKeras.py --train_path inputs/s_by_m_by_pos/dmso.train.hdf5 --valid_path inputs/s_by_m_by_pos/dmso.validate.hdf5 --model_output_file purekeras.NEW.graph.1e-6.hdf5 --model_builder dmso_legacy_graph_model.py --w0_file w0.txt --w1_file w1.txt --batch_size 1000

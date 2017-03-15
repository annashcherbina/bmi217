#python ae_eval.py --dataset /data/deeplearning/multitasked_model/test_data.hdf5 --model optimal_CAE_model.hdf5 --sample 0 --out_prefix autoencoder_eval_sample_0 --plot
for i in `seq 0 93000`
do  
    python ae_eval.py --dataset /srv/scratch/annashch/deeplearning/gecco/inputs/gecco.sampled.one.output/test_data.hdf5 --model optimal_CAE_model.hdf5 --sample $i --out_prefix autoencoder_eval_sample_0 >>accuracies.txt
done    


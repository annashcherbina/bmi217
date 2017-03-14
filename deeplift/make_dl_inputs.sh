#using reference 
#python make_dl_inputs.py --variant_bed pos_of_interest.csv  --seq_size 2000 --ref_file /srv/scratch/annashch/stemcells/het/anna_code/hg19.genome.fa --out_prefix ref.seq.inputs

#with variant 
python make_dl_inputs.py --variant_bed pos_of_interest.csv  --seq_size 2000  --ref_file /srv/scratch/annashch/stemcells/het/anna_code/hg19.genome.fa --out_prefix var.seq.inputs --incorporate_var


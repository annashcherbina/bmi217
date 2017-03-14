import pysam
import h5py
import math
import argparse
import numpy as np
import pdb

def parse_args():
    parser=argparse.ArgumentParser(description='Makes deepLIFT inputs from a list of variant positions')
    parser.add_argument('--variant_bed')
    parser.add_argument('--seq_size',type=int)
    parser.add_argument('--ref_file')
    parser.add_argument('--out_prefix')
    parser.add_argument('--incorporate_var',action='store_true',default=False)
    return parser.parse_args() 

def make_hdf(args,observed_bins):
    total_seqs=len(observed_bins)
    seq_size=args.seq_size 
    outf=h5py.File(args.out_prefix+'.hdf5','w')
    dataset=outf.create_dataset('X/sequence',shape=(1,1,4,seq_size),maxshape=(None,1,4,seq_size))
    ref_source=pysam.FastaFile(args.ref_file)

    #process the first sequence
    observed_bin=observed_bins[0]
    chrom=observed_bin[0]
    pos_start=observed_bin[1]
    pos_end=observed_bin[2] 
    seq = ref_source.fetch(chrom, pos_start, pos_end)
    encoded_seq=one_hot_encode_sequences([seq])
    dataset[:]=encoded_seq
    seq_count=1
    for c in range(1,total_seqs):
        observed_bin=observed_bins[c]
        chrom=observed_bin[0]
        pos_start=observed_bin[1]
        pos_end=observed_bin[2]
        var=observed_bin[3] 
        seq = ref_source.fetch(chrom, pos_start, pos_end)
        if args.incorporate_var:
            varpos=int(math.ceil(args.seq_size/2.0)-1)
            endpos=len(seq)
            seq=seq[0:varpos]+var+seq[varpos+1:endpos]
        encoded_seq=one_hot_encode_sequences([seq])        
        dataset.resize(c+1,axis=0)
        dataset[-1,:,:,:]=encoded_seq
        if c%1000==0:
            print(str(c))
    outf.flush()
    outf.close()

def make_gdl(args,observed_bins):
    outf=open(args.out_prefix+'.gdl','w')
    for i in range(len(observed_bins)):
        outf.write('\t'.join([str(token) for token in observed_bins[i][0:3]])+'\t'+str(i)+'\n')
        
def get_bins(args):
    bin_list=[]
    data=[i.split('\t') for i in open(args.variant_bed,'r').read().strip().split('\n')]
    for entry in data:
        chrom=entry[0]
        if chrom.startswith('chr')==False:
            chrom='chr'+chrom
        pos=int(entry[1])
        flank_low=math.floor(args.seq_size/2.0)
        flank_high=math.ceil(args.seq_size/2.0)
        start=int(pos-flank_low)
        end=int(pos+flank_high)
        if len(entry)<3:
            var=None
        else:
            var=entry[2] 
        bin_list.append([chrom,start,end,var])
    return bin_list

def one_hot_encode_sequences(sequences):
    return np.array([seq_to_2d_image(seq) for seq in sequences])


def seq_to_2d_image(sequence):
    to_return = np.zeros((1,4,len(sequence)), dtype=np.int8)
    seq_to_2d_image_fill_in_array(to_return[0], sequence)
    return to_return


# Letter as 1, other letters as 0
def seq_to_2d_image_fill_in_array(zeros_array, sequence):
    #zeros_array should be an array of dim 4xlen(sequence), filled with zeros.
    #will mutate zeros_array
    for (i,char) in enumerate(sequence):
        if (char=="A" or char=="a"):
            char_idx = 0
        elif (char=="C" or char=="c"):
            char_idx = 1
        elif (char=="G" or char=="g"):
            char_idx = 2
        elif (char=="T" or char=="t"):
            char_idx = 3
        elif (char=="N" or char=="n"):
            continue #leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: "+str(char))
        zeros_array[char_idx,i] = 1

def main():
    args=parse_args()
    bins=get_bins(args)
    #generate hdf5 
    make_hdf(args,bins)
    #generate gdl
    make_gdl(args,bins)

if __name__=="__main__":
    main()
    

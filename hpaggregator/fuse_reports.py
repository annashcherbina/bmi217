import argparse
def parse_args():
    parser=argparse.ArgumentParser("fuse multiple summary files of parameter search")
    parser.add_argument("--inputf",nargs="+",default=[])
    parser.add_argument("--outf")
    return parser.parse_args()

def main():
    args=parse_args()
    fields=set([])
    fused_data=[] 
    for fname in args.inputf:
        data=open(fname,'r').read().strip().split('\n')
        header=data[0].split('\t')
        fields=fields.union(set(header))
        for line in data[1::]:
            cur_dict=dict() 
            tokens=line.split('\t') 
            for i in range(len(header)):
                fieldval=header[i]
                fileval=tokens[i] 
                cur_dict[fieldval]=fileval
            fused_data.append(cur_dict)
    print(str(fused_data))
    print("finished fusion, writing output")

    outf=open(args.outf,'w')
    fields=list(fields)
    outf.write('\t'.join(fields)+'\n')
    for entry in fused_data:
        for field in fields:
            if field in entry:
                outf.write(entry[field]+'\t')
            else:
                outf.write('None\t')
        outf.write('\n')        
if __name__=="__main__":
    main()
    

import sys
diffgenes=open(sys.argv[1],'r').read().strip().split('\n')
diffgene_dict=dict()
for line in diffgenes:
    tokens=line.split('\t')
    gene=tokens[0]
    chrom=tokens[1]
    start=int(tokens[2])
    end=int(tokens[3])
    diffgene_dict[gene]=[chrom,start-5000,end+5000] #take a 5 kb up&down stream flank 
print(str(diffgene_dict))    
vcf=open(sys.argv[2],'r').read().strip().split('\n')
print('read in vcf')
outf=open(sys.argv[3],'w')

for line in vcf:
    if line.startswith('#'):
        continue
    tokens=line.split('\t')[1].split(':')
    chrom=tokens[0]
    pos=int(tokens[1])
    for gene in diffgene_dict:
        if chrom==diffgene_dict[gene][0]:
            if (pos >=diffgene_dict[gene][1]):
                if (pos <=diffgene_dict[gene][2]):
                    outf.write(gene+'\t'+line+'\n')
                    print str(line)
                    

source=open('intersection.txt','r').read().strip().split('\n')
intersection_dict=dict()
for line in source:
    tokens=line.split('\t')
    entry=tuple([tokens[0],tokens[1]])
    intersection_dict[entry]=1
outf=open('intersection.annotated','w')
snps=open('SNPs.vcf','r')
for line in snps:
    if line.startswith('#'):
        continue
    tokens=line.split('\t')
    entry=tuple([tokens[0],tokens[1]])
    if entry in intersection_dict:
        outf.write(line+'\n')
        

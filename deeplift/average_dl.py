import sys 
data=open(sys.argv[1],'r').read().strip().split('\n') 
outf=open(sys.argv[2],'w',100000) 

base_dict=dict() 
counter=0 
for line in data: 
    counter+=1 
    if counter%1000000==0: 
        print str(counter) 
    tokens=line.split('\t') 
    chrom=tokens[0] 
    if len(tokens)<2: 
        continue 
    try:
        startbase=int(tokens[1]) 
    except: 
        print str(tokens) 
    endbase=tokens[2] 
    '''
    content=tokens[-1].split(',') 
    idval=content[0].split(':')[-1] 
    score=content[1].split('[')[-1] 
    color=content[2].split(']')[0] 
    '''
    score=tokens[-1] 
    if float(score)<1e-3: 
        continue 
    #print str(score) 
    if chrom not in base_dict: 
        base_dict[chrom]=dict() 
    if startbase not in base_dict[chrom]: 
        base_dict[chrom][startbase]=dict() 
        base_dict[chrom][startbase]['endbase']=endbase 
        #base_dict[startbase]['id']=idval 
        base_dict[chrom][startbase]['score']=[score] 
        #base_dict[startbase]['color']=color 
    else: 
        base_dict[chrom][startbase]['score'].append(score) 
print "performing averaging" 
#perform the average 
for chrom in base_dict: 
    for startbase in base_dict[chrom]: 
        base_dict[chrom][startbase]['score']=sum([float(i) for i in base_dict[chrom][startbase]['score']])/len(base_dict[chrom][startbase]['score'])
print "writing output" 
#write the average hammock file 
uniquechroms=base_dict.keys() 
uniquechroms.sort() 
for chrom in uniquechroms: 
    uniquebases=base_dict[chrom].keys() 
    uniquebases.sort() 
    for base in uniquebases: 
        curentry=base_dict[chrom][base] 
        #chrom=curentry['chrom'] 
        endbase=curentry['endbase'] 
        #idval=curentry['id'] 
        score=curentry['score'] 
        #color=curentry['color'] 
        #outf.write(chrom+'\t'+str(base)+'\t'+str(endbase)+'\t'+'id:'+str(idval)+','+'qcat:[['+str(score)+','+str(color)+']]\n')
        outf.write(chrom+'\t'+str(base)+'\t'+str(endbase)+'\t'+str(score)+'\n')

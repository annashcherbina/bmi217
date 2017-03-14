#hacky script to generate a venn diagram from SIFT, polyphen, CADD results
import operator

#load the data
sift389=open('v389.sift','r').read().strip().split('\n')
polyphen389=open('v389.polyphen','r').read().strip().split('\n')
sift410=open('v410.sift','r').read().strip().split('\n')
polyphen410=open('v410.polyphen','r').read().strip().split('\n')
sift576=open('v576.sift','r').read().strip().split('\n')
polyphen576=open('v576.polyphen','r').read().strip().split('\n')
#cadd_high=open('highest1000CADD.txt','r').read().strip().split('\n')
cadd_high=open("caddOver20.txt",'r').read().strip().split('\n')

#create dictionaries of subjects:
v389_dict=dict()
v410_dict=dict()
v576_dict=dict()

for line in sift389:
    tokens=line.split('\t')
    var=tokens[1]
    v389_dict[var]=set(['sift'])
for line in polyphen389:
    tokens=line.split('\t')
    var=tokens[1]
    if var in v389_dict:
        v389_dict[var].add('polyphen')
    else:
        v389_dict[var]=set(['polyphen'])

for line in cadd_high:
    tokens=line.split('\t')
    var=tokens[0]+':'+tokens[1]
    if var in v389_dict:
        v389_dict[var].add('cadd')
    else:
        v389_dict[var]=set(['cadd'])

for line in sift410:
    tokens=line.split('\t')
    var=tokens[1]
    v410_dict[var]=set(['sift'])
for line in polyphen410:
    tokens=line.split('\t')
    var=tokens[1]
    if var in v410_dict:
        v410_dict[var].add('polyphen')
    else:
        v410_dict[var]=set(['polyphen'])

for line in cadd_high:
    tokens=line.split('\t')
    var=tokens[0]+':'+tokens[1]
    if var in v410_dict:
        v410_dict[var].add('cadd')
    else:
        v410_dict[var]=set(['cadd'])

for line in sift576:
    tokens=line.split('\t')
    var=tokens[1]
    v576_dict[var]=set(['sift'])
for line in polyphen576:
    tokens=line.split('\t')
    var=tokens[1]
    if var in v576_dict:
        v576_dict[var].add('polyphen')
    else:
        v576_dict[var]=set(['polyphen'])

for line in cadd_high:
    tokens=line.split('\t')
    var=tokens[0]+':'+tokens[1]
    if var in v576_dict:
        v576_dict[var].add('cadd')
    else:
        v576_dict[var]=set(['cadd'])

#get the venn counts!
print("v389:")
v389_summary=dict()
for var in v389_dict:
    entry=tuple(v389_dict[var])
    if entry not in v389_summary:
        v389_summary[entry]=1
    else:
        v389_summary[entry]+=1
for key in v389_summary:
    print str(key)+":"+str(v389_summary[key])
    

print("v410:")
v410_summary=dict()
for var in v410_dict:
    entry=tuple(v410_dict[var])
    if entry not in v410_summary:
        v410_summary[entry]=1
    else:
        v410_summary[entry]+=1
for key in v410_summary:
    print str(key)+":"+str(v410_summary[key])

print("v576:")
v576_summary=dict()
for var in v576_dict:
    entry=tuple(v576_dict[var])
    if entry not in v576_summary:
        v576_summary[entry]=1
    else:
        v576_summary[entry]+=1
for key in v576_summary:
    print str(key)+":"+str(v576_summary[key])
    


        
#get the intersection set!
allkeys=v389_dict.keys()+v410_dict.keys()+v576_dict.keys()
allkeys=set(allkeys)
intersection_dict=dict()
for key in allkeys:
    numhits=0
    support=set([])
    if key in v389_dict:
        numhits+=1
        support=support.union(v389_dict[key])
    if key in v410_dict:
        numhits+=1
        support=support.union(v410_dict[key])
    if key in v576_dict:
        numhits+=1
        support=support.union(v576_dict[key])
    if numhits==3:
        intersection_dict[key]=str(support)


        
sorted_intersection=sorted(intersection_dict.items(),key=operator.itemgetter(1))
sorted_intersection.reverse()
outf=open('intersection.txt','w')
bar_graph=dict() 
for entry in sorted_intersection:
    outf.write(str(entry[0])+'\t'+str(entry[1])+'\n')
    num_evidence=entry[1]
    if num_evidence not in bar_graph:
        bar_graph[num_evidence]=1
    else:
        bar_graph[num_evidence]+=1
outf=open('bargraph.txt','w')
for entry in bar_graph:
    outf.write(str(entry)+'\t'+str(bar_graph[entry])+'\n')
    


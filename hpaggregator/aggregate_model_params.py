#This script aggregates a list of hyperparameters across multiple models into a csv flat file
import argparse
import json
import h5py
import yaml
import pdb
from collections import OrderedDict

def parse_args():
    parser=argparse.ArgumentParser(description="This script aggregates a list of hyperparameters across multiple models into a csv flat file")
    parser.add_argument("--yaml",nargs="+",default=[])
    parser.add_argument("--model_hdf5",nargs="+",default=[])
    parser.add_argument("--fields_to_include",help="file containing field names to include for summarizing")
    parser.add_argument("--outf")
    parser.add_argument("--result_json",nargs="*",default=[])
    parser.add_argument("--result_csv",nargs="*",default=[])
    parser.add_argument("--result_fields_json",nargs="*",default=[])
    parser.add_argument("--result_fields_csv",nargs="*",default=[]) 
    return parser.parse_args()


def get_architecture_hdf5(args):
    configs=[] 
    for model_hdf5 in args.model_hdf5:
        model_hdf5=h5py.File(model_hdf5)
        model_config=model_hdf5.attrs.get('model_config')
        model_config=json.loads(model_config.decode('utf-8'))
        configs.append(model_config)
    return configs
    

def get_architecture_yaml(args):
    configs=[] 
    for cur_yaml in args.yaml:
        yaml_string=open(cur_yaml,'r').read()
        model_config=yaml.load(yaml_string)
        
        #need to order the nodes appropriately
        ordered_config=OrderedDict()
        chain=dict()
        first=None

        nodes=model_config['config']['nodes']
        node_config=model_config['config']['node_config']
        for entry in node_config:
            node_name=entry['name']
            node_input=entry['input']
            if node_input=='sequence':
                first=node_name
            chain[node_input]=node_name
        ordered_names=[]
        while first in chain:
            ordered_names.append(first)
            first=chain[first]
        #iterate through the ordered names 
        for entry in ordered_names:
            ordered_config[entry]=nodes[entry]
        configs.append(ordered_config)
    return configs

#recursively iterate a json 
def recurse(data,fields_to_include,parent,aggregate_elems):
    if (type(data) is dict)or (type(data) is OrderedDict): 
        for element in data.items():
            if element[0] in fields_to_include:
                #we care about this element!
                #get the full path
                elem_path=parent+[element[0]]
                value=element[1]
                final_key=element[0]
                path_length=len(elem_path)
                key_to_store=final_key#tuple([final_key,path_length])
                if key_to_store not in aggregate_elems:
                    aggregate_elems[key_to_store]=[value]
                else:
                    aggregate_elems[key_to_store].append(value)
            if type(element[1]) is list or type(element[1]) is dict or type(element[1]) is OrderedDict:
                recurse(element[1],fields_to_include,parent+[element[0]],aggregate_elems)
    elif type(data)==list:
        for element in data:
            recurse(element,fields_to_include,parent,aggregate_elems)
    return aggregate_elems

#map results to model architectures 
def parse_results_json(args):
    fields=args.result_fields_json
    dbs=args.result_json
    results=dict() 
    for db in dbs:
        data=json.loads(open(db,'r').read())['records']
        for record in data:
            yaml_name=record["saved_files_config"]['yaml_file'].split('/')[-1]
            perf=record['best_valid_perf_info']['valid_all_stats']
            performance=[]
            for field in fields:
                if field in perf:
                    performance.append(perf[field])
                else:
                    performance.append(None) 
            results[yaml_name]=performance
    return results 

#map csv results to model architectures
def parse_results_csv(args):
    fields=args.result_fields_csv
    dbs=args.result_csv
    results=dict()
    for db in dbs:
        data=open(db,'r').read().strip().split('\n')
        model_name=db.split('/')[-1]
        performance=dict()
        for line in data:
            tokens=line.split('\t')
            metric_name=tokens[0]
            metric_vals=[float(i) for i in tokens[2:]]
            metric_mean=sum(metric_vals)/len(metric_vals)
            performance[metric_name]=metric_mean
        results[model_name]=[]
        for field in fields:
            results[model_name].append(performance[metric_name])
    return results

def main():
    args=parse_args()
    fields_to_include=open(args.fields_to_include,'r').read().strip().split('\n')
    fields_to_include_dict=dict()
    for field in fields_to_include:
        fields_to_include_dict[field]=1
    print('got dictionary of fields to include')
    yaml_models=[]
    hdf5_models=[]
    model_names=args.yaml+args.model_hdf5
    if len(args.yaml)>0:
        yaml_models=get_architecture_yaml(args)
    if len(args.model_hdf5) >0:
        hdf5_models=get_architecture_hdf5(args)
    all_models=yaml_models+hdf5_models
    print("read in all model architectures")

    #dictionary to store aggregate fields of interest
    field_dict=dict()
    all_fields=set([]) 
    for i in range(len(all_models)):
        cur_model_name=model_names[i].split('/')[-1]
        cur_model=all_models[i]
        field_dict[cur_model_name]=recurse(cur_model,fields_to_include,[],dict())
        #summarize the count of each type of class
        class_names=field_dict[cur_model_name]['class_name']
        class_counts=dict()
        for entry in class_names:
            if entry not in class_counts:
                class_counts[entry]=1
            else:
                class_counts[entry]+=1
        for entry in class_counts:
            field_dict[cur_model_name][entry]=class_counts[entry]
    
        new_fields=set(field_dict[cur_model_name].keys())
        all_fields=all_fields.union(new_fields)

    #add in any known info about the results
    if (len(args.result_json)>0) and (len(args.result_fields_json)>0):
        result_dict_json=parse_results_json(args)
    else:
        result_dict_json=dict() 

    if (len(args.result_csv)>0) and (len(args.result_fields_csv)>0):
        result_dict_csv=parse_results_csv(args)
    else:
        result_dict_csv=dict()

    result_dict_json.update(result_dict_csv)
    result_dict=result_dict_json
    
    #write the output
    all_fields=list(all_fields)
    if len(args.result_fields_json)>0:
        all_fields=all_fields+args.result_fields_json
    if len(args.result_fields_csv)>0:
        all_fields=all_fields+args.result_fields_csv
    
    outf=open(args.outf,'w')
    outf.write('Model\t'+'\t'.join([str(i) for i in all_fields])+'\n')
    for model in field_dict:
        outf.write(model)
        for field in all_fields:
            if field in field_dict[model]:
                outf.write('\t'+str(field_dict[model][field]))
            else:
                outf.write('\tNone')
        if model in result_dict:
            outf.write('\t'+'\t'.join([str(i) for i in result_dict[model]]))
        outf.write('\n')
        
    
        

if __name__=="__main__":
    main()
    

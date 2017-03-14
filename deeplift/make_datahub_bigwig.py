import pdb 
import argparse
import glob
from itertools import product
import json
import os
import urlparse

# NOTE: JSON requires double-quotes; single-quotes are not valid
BIGWIG_JSON = """
    {
        "colorpositive":"#0000ff",
        "mode": "show",
        "name": "",
        "qtc": {
            "anglescale": 1,
            "height": 40,
            "pb": 128,
            "pg": 0,
            "pr": 0,
            "smooth": 3,
            "summeth": 2,
            "thtype": 0
        },
        "type": "bigwig",
        "height":20,
        "url": ""
    }
"""


QCAT_SEQUENCE_JSON = """
    {
        "type": "quantitativeCategorySeries",
        "name": "",
        "height": 64,
        "url": "",
        "backgroundcolor": "#ffffff",
        "mode": "show",
        "categories": {
              "0": ["A", "#ff0000"],
              "1": ["C", "#0000ff"],
              "2": ["G", "#ffa500"],
              "3": ["T", "#00ff00"]
        }
    }
"""


QCAT_MOTIF_JSON = """
    {
        "type": "hammock",
        "name": "",
        "height": 64,
        "url": "",
        "backgroundcolor": "#ffffff",
        "mode": "barplot",
        "scorenamelst":["Corr."],
        "scorescalelst":[{"type":0}],
        "showscoreidx":0
    }
"""



QCAT_COMBINED_JSON = """
    {
        "type": "quantitativeCategorySeries",
        "name": "",
        "height": 64,
        "url": "",
        "backgroundcolor": "#ffffff",
        "mode": "show",
        "categories": {
              "0":["dnase", "#c0c0c0"],
              "1":["mnase", "#606060"],
              "2":["sequence", "#000099"]
        }
    }
"""
BEDGRAPH_JSON="""
    {
        "height": 50, 
        "type": "bedGraph",
        "fixedscale":{"min": -1,"max": 1},
        "url": "",
        "name": "",
        "mode": "show"
    }

"""
'''
name_dict=dict() 
name_dict[0]='CC Peak Presenece'
name_dict[1]='3hr Peak Presence' 
name_dict[2]='16hr Peak Presence' 
name_dict[3]='48hr Peak Presence' 
name_dict[4]='H1 Peak Presence' 
name_dict[5]='Hk Peak Presence' 
name_dict[6]='M5 Peak presence' 
name_dict[7]='3hr Upregulate2d from CC' 
name_dict[8]='3hr Downregulated from CC' 
name_dict[9]='16hr Upregulated from 3hr' 
name_dict[10]='16hr Downregulated from 3hr' 
name_dict[11]='16hr Upregulated from CC' 
name_dict[12]='16hr Downregulated from CC'
name_dict[13]='48hr Upregulated from 16hr' 
name_dict[14]='48hr Downregulated from 16hr'
name_dict[15]='48hr Upregulated from 3hr' 
name_dict[16]='48hr Downregulated from 3hr'
name_dict[17]='48hr Upregulated from CC' 
name_dict[18]='48hr Downregulated from CC'
name_dict[19]='H1 Upregulated from 48hr' 
name_dict[20]='H1 Downregulated from 48hr'
name_dict[21]='H1 Upregulated from 16hr' 
name_dict[22]='H1 Downregulated from 16hr'
name_dict[23]='H1 Upregulated from 3hr' 
name_dict[24]='H1 Downregulated from 3hr'
name_dict[25]='H1 Upregulated from CC' 
name_dict[26]='H1 Downregulated from CC'
name_dict[27]='H1 Upregulated from Hk' 
name_dict[28]='H1 Downregulated from Hk' 
name_dict[29]='48hr Upregulated from Hk' 
name_dict[30]='48hr Downregulated from Hk' 
name_dict[31]='16hr Upregulated from Hk' 
name_dict[32]='16hr Downregulated from Hk' 
name_dict[33]='3hr Upregulated from Hk' 
name_dict[34]='3hr Downregulated from Hk'
name_dict[35]='H1 Upregulated from M5' 
name_dict[36]='H1 Downregulated from M5' 
name_dict[37]='48hr Upregulated from M5' 
name_dict[38]='48hr Downregulated from M5' 
name_dict[39]='16hr Upregulated from M5' 
name_dict[40]='16hr Downregulated from M5' 
name_dict[41]='3hr Upregulated from M5' 
name_dict[42]='3hr Downregulated from M5' 
'''

def parse_args():
    # TODO: add BASE_URL and BASE_DATAHUB as actual options
    parser = argparse.ArgumentParser(
        description='Make Datahub JSONs for DeepLIFT tracks')

    parser.add_argument('tracks_dir',
                        help='Directory where all the trackfiles are '
                             ' (in subdirectories; see --singledir '
                             '  if this is the target directory)')

    parser.add_argument('--singledir',
                        help='tracks_dir directly contains the files',
                        action='store_true')
    parser.add_argument('--copy',
                        help='Also copy bigwig and hammock files '
                             'to destination directory.')
    parser.add_argument('--motifDir',
                        help='directory where the motif names are ')
    parser.add_argument('--trueAnnotationDir',
                        help='directory containing the true annotation files',default=None)
    parser.add_argument('--predictedAnnotationDir',
                        help='directory containing the predicted values',default=None)
    parser.add_argument('--nameListFile',
                        help='List of Task Names',default=None)

    args = parser.parse_args()

    BASE_DATAHUB = 'viz-dl.details.json'
    BASE_URL = 'http://mitra.stanford.edu/kundaje/annashch/het/'

    setattr(args, 'base_datahub', BASE_DATAHUB)
    setattr(args, 'base_url', BASE_URL)

    return args


def make_datahub_from_directory(directory, base_datahub, base_url,
                                output_file,args):
    #MARKS = ['h3k4me1', 'h3k4me3', 'h3k27ac', 'ctcf','het']
    motifDir=args.motifDir
    trueAnnotationDir=args.trueAnnotationDir 
    predictedAnnotationDir=args.predictedAnnotationDir 

    MARKS=['anna.positional'] 
    INPUTS=[str(i) for i in range(61)]
    #INPUTS = ['atac-dnase', 'atac-mnase', 'atac-sequence','0']
    TRACKS = ['DeepLIFT']
    input2json = {
        'atac-dnase': BIGWIG_JSON,
        'atac-mnase': BIGWIG_JSON,
        'atac-sequence': QCAT_SEQUENCE_JSON,
        'combined': QCAT_COMBINED_JSON,
        '0': BIGWIG_JSON,
    }

    input2ext = {
        'atac-dnase': 'bw',
        'atac-mnase': 'bw',
        'atac-sequence': 'gz',
        'combined': 'gz',
        '0': 'gz',
    }

    datahub_json = json.load(open(base_datahub))
    datahub_name = os.path.basename(directory)

    for mark, track, input_type in product(MARKS, TRACKS, INPUTS):
        #ext = input2ext[input_type]
        ext='bigWig'
        motif_ext='hammock.sorted.gz'
        track_files = sorted(glob.glob(
            os.path.join(directory,
                         '{}_{}_{}*.{}'.format(mark, input_type, track, ext))))
        if motifDir!=None: 
            motif_files=sorted(glob.glob(
                os.path.join(motifDir,
                             '{}_{}_{}*.{}'.format(mark,input_type,'deepLIFT',motif_ext))))
        else: 
            motif_files=[] 
        if trueAnnotationDir!=None: 
            true_annotation_files=sorted(glob.glob(
                os.path.join(trueAnnotationDir,
                             'true.{}.sorted.bedGraph.gz'.format(str(int(input_type))))))
            #print str(os.path.join(trueAnnotationDir,'true.{}.bedGraph.gz'.format(str(int(input_type)+4))))
            #pdb.set_trace() 
        else: 
            true_annotations_files=[] 
        
        if predictedAnnotationDir!=None: 
            predicted_annotation_files=sorted(glob.glob(
                os.path.join(predictedAnnotationDir,
                             'predicted.{}.sorted.bedGraph.gz'.format(str(int(input_type))))))
        else: 
            predicted_annotation_files=[] 

        print str(mark) 
        print str(input_type) 
        print str(track) 
        print str(ext) 
        '''
        try:
            #pdb.set_trace() 
            track_files = [track_files[0]]
        except IndexError:
            print('Warning: {} is missing files for mark {} input_type {}'
                  ' track {}'.format(directory, mark, input_type, track))
        '''
        for i in range(len(track_files)):
            #track_json = json.loads(input2json[input_type])
            track_file=track_files[i] 
            track_json = json.loads(BIGWIG_JSON)
            input_name = (input_type[5:] if input_type.startswith('atac-')
                          else input_type)
            #pdb.set_trace() 
            track_json['name']=name_dict[int(input_name)]
            #track_json['metadata']=metadata_dict[input_name] 
            track_json['url'] = urlparse.urljoin(base_url, track_file)
            track_json['fixedscale']=dict() 
            track_json['fixedscale']['min']=0.005
            track_json['fixedscale']['max']=0.1 
            
            datahub_json.append(track_json)
            #add in the true positive & predicted tracks 
            '''
            if trueAnnotationDir: 
                true_annotation_file=true_annotation_files[i] 
                true_json=json.loads(BEDGRAPH_JSON) 
                input_name=(input_type[5:] if input_type.startswith('atac-')
                            else input_type) 
                true_json['name']='True:'+name_dict[int(input_name)]
                true_json['metadata']=metadata_dict['1000'] 
                true_json['url']=urlparse.urljoin(base_url+'true_annotation_dir',true_annotation_file)
                datahub_json.append(true_json) 
            if predictedAnnotationDir:                 
                predicted_annotation_file=predicted_annotation_files[i] 
                predicted_json=json.loads(BEDGRAPH_JSON) 
                input_name=(input_type[5:] if input_type.startswith('atac-') 
                            else input_type) 
                predicted_json['name']='Predicted:'+name_dict[int(input_name)] 
                predicted_json['metadata']=metadata_dict['1001'] 
                predicted_json['url']=urlparse.urljoin(base_url+'predicted_annotation_dir',predicted_annotation_file) 
                datahub_json.append(predicted_json) 
            '''
            #pdb.set_trace()
            #get the associated motif json! 
            if motifDir:
                motif_file=motif_files[i] 
                motif_json=json.loads(QCAT_MOTIF_JSON) 
                input_name=(input_type[5:] if input_type.startswith('atac-')
                            else input_type) 
                motif_json['name']='Motifs:'+name_dict[int(input_name)]
                #motif_json['metdata']=metadata_dict[input_name] 
                motif_json['url']=urlparse.urljoin(base_url,motif_file)
                datahub_json.append(motif_json) 
            
    for mark, track, input_type in product(MARKS, TRACKS, INPUTS):
        #ext = input2ext[input_type]
        ext='bigWig'
        motif_ext='hammock.sorted.gz'
        track_files = sorted(glob.glob(
            os.path.join(directory,
                         '{}_{}_{}*.{}'.format(mark, input_type, track, ext))))
        if motifDir!=None: 
            motif_files=sorted(glob.glob(
                os.path.join(motifDir,
                             '{}_{}_{}*.{}'.format(mark,input_type,'deepLIFT',motif_ext))))
        else: 
            motif_files=[] 
        if trueAnnotationDir!=None: 
            true_annotation_files=sorted(glob.glob(
                os.path.join(trueAnnotationDir,
                             'true.{}.sorted.bedGraph.gz'.format(str(int(input_type))))))
            #print str(os.path.join(trueAnnotationDir,'true.{}.bedGraph.gz'.format(str(int(input_type)+4))))
            #pdb.set_trace() 
        else: 
            true_annotations_files=[] 
        
        if predictedAnnotationDir!=None: 
            predicted_annotation_files=sorted(glob.glob(
                os.path.join(predictedAnnotationDir,
                             'predicted.{}.sorted.bedGraph.gz'.format(str(int(input_type))))))
        else: 
            predicted_annotation_files=[] 

        print str(mark) 
        print str(input_type) 
        print str(track) 
        print str(ext) 
        '''
        try:
            #pdb.set_trace() 
            track_files = [track_files[0]]
        except IndexError:
            print('Warning: {} is missing files for mark {} input_type {}'
                  ' track {}'.format(directory, mark, input_type, track))
        '''
        for i in range(len(track_files)):
            '''
            #track_json = json.loads(input2json[input_type])
            track_file=track_files[i] 
            track_json = json.loads(BIGWIG_JSON)
            input_name = (input_type[5:] if input_type.startswith('atac-')
                          else input_type)
            track_json['name']=name_dict[int(input_name)]
            track_json['metadata']=metadata_dict[input_name] 
            track_json['url'] = urlparse.urljoin(base_url, track_file)
            datahub_json.append(track_json)
            #add in the true positive & predicted tracks 
            '''
            if trueAnnotationDir: 
                true_annotation_file=true_annotation_files[i] 
                true_json=json.loads(BEDGRAPH_JSON) 
                input_name=(input_type[5:] if input_type.startswith('atac-')
                            else input_type) 
                true_json['name']='True:'+name_dict[int(input_name)]
                #true_json['metadata']=metadata_dict['1000'] 
                true_json['url']=urlparse.urljoin(base_url+'true_annotation_dir',true_annotation_file)
                datahub_json.append(true_json) 
            if predictedAnnotationDir:                 
                predicted_annotation_file=predicted_annotation_files[i] 
                predicted_json=json.loads(BEDGRAPH_JSON) 
                input_name=(input_type[5:] if input_type.startswith('atac-') 
                            else input_type) 
                predicted_json['name']='Predicted:'+name_dict[int(input_name)] 
                #predicted_json['metadata']=metadata_dict['1001'] 
                predicted_json['url']=urlparse.urljoin(base_url+'predicted_annotation_dir',predicted_annotation_file) 
                datahub_json.append(predicted_json) 

            #pdb.set_trace()
            #get the associated motif json! 
            if motifDir:
                motif_file=motif_files[i] 
                motif_json=json.loads(QCAT_MOTIF_JSON) 
                input_name=(input_type[5:] if input_type.startswith('atac-')
                            else input_type) 
                motif_json['name']='Motifs:'+name_dict[int(input_name)]
                #motif_json['metdata']=metadata_dict[input_name] 
                motif_json['url']=urlparse.urljoin(base_url,motif_file)
                datahub_json.append(motif_json) 
                
    full_output_file = os.path.join(directory, output_file)
    json.dump(datahub_json, open(full_output_file, 'w'), indent=4)
    return (datahub_name, full_output_file)


def make_index_page(datahubs, base_url, output_file):
    BROWSER_BASE_URL = \
        'http://epigenomegateway.wustl.edu/browser/?genome=hg19&datahub={}&tknamewidth=150'

    with open(output_file, 'w') as fp:
        fp.write('<html>\n')
        fp.write('<head>\n')
        fp.write('<title>DeepLIFT tracks</title>\n')
        fp.write('</head>\n')
        fp.write('<body>\n')
        fp.write('<ul>\n')
        for datahub_name, datahub_path in datahubs:
            datahub_url = urlparse.urljoin(base_url, datahub_path)
            browser_url = BROWSER_BASE_URL.format(datahub_url)
            fp.write('<li><a href="{}">{}</a></li>\n'.format(browser_url, datahub_name))
        fp.write('</ul>\n')
        fp.write('</body>\n')
        fp.write('</html>\n')

def make_name_dict(nameListFile):
    name_dict=dict()
    data=open(nameListFile,'r').read().strip().split('\n')
    for i in range(len(data)):
        name_dict[i]=data[i] 
    return name_dict
        
if __name__ == '__main__':
    args = parse_args()
    name_dict=make_name_dict(args.nameListFile)
    dirs_to_process = \
        ([args.tracks_dir] if args.singledir else
         [os.path.join(args.tracks_dir, subdir)
          for subdir in sorted(next(os.walk(args.tracks_dir))[1])])
    print str(dirs_to_process) 
    for i in range(len(dirs_to_process)): 
        make_datahub_from_directory(dirs_to_process[i],args.base_datahub,args.base_url,'viz-dl.details.bigwig.json',args)

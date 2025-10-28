############################################################################
### This script can be used for loading various information of dataset such
### as image path, named entities present in the image, their qids, questions 
### about image, paraphrased questions, corresponding answer, train/test/
### validation, type of question, and splits. Author: Anand Mishra, 
### Email: anandmishra@iisc.ac.in 
############################################################################
import json
import sys
import pdb
with open('dataset.json', 'r') as fp:
        data = json.load(fp)

pdb.set_trace()
for k in data.keys():
    print(data[k]['imgPath'])
    print(data[k]['Qids'])
    print(data[k]['NamedEntities'])
    print(data[k]['wikiCap'])
    print(data[k]['Questions'])
    print(data[k]['ParaQuestions'])
    print(data[k]['Answers'])
    print(data[k]['Type of Question'])
    print(data[k]['split'])
    print('\n***************************\n')
    pdb.set_trace()


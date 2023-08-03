import json
import os
from os import path
import shutil
from tqdm import tqdm
from PIL import Image

CURR_PATH = path.dirname(__file__)
DST_FOLDER = "/root/notebooks/nfs/work/barry.chen/DenoisingDiffusionProbabilityModel-ddpm-/dataset/test_all"
SRC_FOLDER = "/root/notebooks/nfs/work/dataset/Phison/datasets_for_ma"



def MakeSubsetImg(conditions: list, components: list, n = 0):

    if not os.path.exists(DST_FOLDER):
            os.makedirs(DST_FOLDER)
    
    with open(path.join(SRC_FOLDER, "dataset.json"), 'r') as j:    
        attr_dict = {} # to record each img's attribute from json label only
        json_arr = []
        class_cnt = [0 for i in range(len(conditions) * len(components))]
        
        json_obj = json.load(j)
    
        for img_name in tqdm(json_obj):
            attr_dict = json_obj[img_name] # run through whole dataset label, 
             
            if(attr_dict['class'] in conditions and attr_dict['component_name'] in components):
                #print(attr_dict['class'],", ",attr_dict['component_name'])
                #print("img name:",img_name)
                i = conditions.index(attr_dict['class']) + 1
                j = components.index(attr_dict['component_name']) * len(conditions)
                
                if class_cnt[j+i-1] >= n and n > 0:
#                     print(attr_dict['component_name'], "+", attr_dict['class'], "reached", n)
                    continue #to next img_name
                
                try:
                    subdict = {"file_name": "", "text": ""}
                    subdict["file_name"] = img_name
                    subdict["text"] = attr_dict["class"] + " " + attr_dict["component_name"]
                    
                    json_arr.append(subdict)
                    

                    shutil.copy(os.path.join(SRC_FOLDER, img_name), DST_FOLDER)
                    class_cnt[j+i-1] += 1

                except:
                    print(img_name, "doesn't exist")
                    json_dict['labels'].pop()
                    class_cnt[j+i-1] -= 1
            
        subdumps = json.dumps(json_arr, sort_keys=True)    
        
        with open(path.join(DST_FOLDER, "metadata.jsonl"), 'w') as outfile: 
            outfile.write(subdumps)
    return




if __name__ == "__main__":
    
    # first choose attributes we need from whole dataset
    # then copy the images that fit the the attribute to dst folder
    # make new json file for sub dataset
    
    #conditions = ['good', 'broke', 'shift', 'missing', "short", 'stand']
    #components = ['C0201', 'C0402', 'C0603', 'C0604', 'R0201', 'R0402', 'R0805', 'L2016', 'LED0603', 'D0603', 'SOT23', 'F1210']
    
    condition1 = ['good', 'broke', 'shift', "short"]
    component1 = ['C0201', 'SOT23', 'L2016', 'C0604', 'R0402', 'C0402']

    MakeSubsetImg(condition1, component1, 20)






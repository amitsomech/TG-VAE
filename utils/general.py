###############################
####General Utils        ######
###############################

import uuid
import os
#import utils.config
from collections import Counter
BASE_DIR = "models/"

def gen_model_uuid(label=None):
    uu = uuid.uuid4().hex
    #os.path.join(
    #os.mkdir(uu)
    if label:
        return f"{uu}_{label}"
    else:
        return uu


def gen_dir(model_uuid,added):
#    base_dir=utils.config.BASE_DIR
    base_dir = BASE_DIR
    new_dir = os.path.join(base_dir,model_uuid,added)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    return new_dir
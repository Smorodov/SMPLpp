# Copyright 2018 Chongyi Zheng. All rights reserved.
#
# This software implements a 3D human skinning model, SMPL, with tensorflow
# and numpy.
# For more detail, see the paper - SMPL: A Skinned Multi-Person Linear Model -
# published by Max Planck Institute for Intelligent Systems on SIGGRAPH ASIA 2015.
#
# Here we provide the software for research purposes only.
# More information about SMPL is available on http://smpl.is.tue.mpg.
#
# ============================= preprocess.py =================================
# File Description:
#
# This file loads the models downloaded from the official SMPL website, grab
# data and write them in to numpy and json format.
#
# =============================================================================
#!/usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import pickle as pkl



def main(args):
    """Main entrance.

    Arguments
    ----------
    - args: list of strings
        Command line arguments.

    Returns
    ----------

    """
    modelName="MANO_left"
    #modelName="SMPLH_female"
    raw_model_path = modelName+'.pkl'
    save_dir = 'result'


    NP_SAVE_FILE = modelName+'.npz'        

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np_save_path = os.path.join(save_dir, NP_SAVE_FILE)   
    '''
     * Model Data Description * #
     vertices_template: global vertex locations of template - (6890, 3)
     face_indices: vertex indices of each face (triangles) - (13776, 3)
     joint_regressor: joint regressor - (24, 6890)
     kinematic_tree_table: table of kinematic tree - (2, 24)
     weights: weights for linear blend skinning - (6890, 24)
     shape_blend_shapes: shape blend shapes - (6890, 3, 10)
     pose_blend_shapes: pose blend shapes - (6890, 3, 207)
     * Extra Data Description *
     Besides the data above, the official model provide the following things.
     The pickle file downloaded from SMPL website seems to be redundant or
     some of the contents are used for training the model. None of them will
     be used to generate a new skinning.
    
     bs_stype: blend skinning style - (default)linear blend skinning
     bs_type: blend skinning type - (default) linear rotation minimization
     J: global joint locations of the template mesh - (24, 3)
     J_regressor_prior: prior joint regressor - (24, 6890)
     pose_training_info: pose training information - string list with 6
                         elements.
     vert_sym_idxs: symmetrical corresponding vertex indices - (6890, )
     weights_prior: prior weights for linear blend skinning
    '''
    
    with open(raw_model_path, 'rb') as f:
        raw_model_data = pkl.load(f,encoding='latin1')
        
    vertices_template = np.require(raw_model_data['v_template'],dtype=np.float32,requirements=['C'])
    face_indices = np.require((raw_model_data['f'] + 1),dtype=np.int32,requirements=['C'])  # starts from 1
    weights = np.require(raw_model_data['weights'],dtype=np.float32,requirements=['C'])
    shape_blend_shapes = np.require(raw_model_data['shapedirs'],dtype=np.float32,requirements=['C'])
    pose_blend_shapes = np.require(raw_model_data['posedirs'],dtype=np.float32,requirements=['C'])
    joint_regressor = np.require(raw_model_data['J_regressor'].toarray(),dtype=np.float32,requirements=['C'])
    kinematic_tree = np.require(raw_model_data['kintree_table'],dtype=np.int32,requirements=['C'])
    
    print(kinematic_tree)
    
    model_data_np = {
        'vertices_template': vertices_template,
        'face_indices': face_indices,
        'weights': weights,
        'shape_blend_shapes': shape_blend_shapes,
        'pose_blend_shapes': pose_blend_shapes,
        'joint_regressor': joint_regressor,
        'kinematic_tree': kinematic_tree
    }
    np.savez(np_save_path, **model_data_np)
    print('Save SMPL Model to: ', os.path.abspath(save_dir))


if __name__ == '__main__':
    #if sys.version_info[0] != 2:
    #    raise EnvironmentError('Run this file with Python2!')

    main(sys.argv)

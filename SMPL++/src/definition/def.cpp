/* ========================================================================= *
 *                                                                           *
 *                                 SMPL++                                    *
 *                    Copyright (c) 2018, Chongyi Zheng.                     *
 *                          All Rights reserved.                             *
 *                                                                           *
 * ------------------------------------------------------------------------- *
 *                                                                           *
 * This software implements a 3D human skinning model - SMPL: A Skinned      *
 * Multi-Person Linear Model with C++.                                       *
 *                                                                           *
 * For more detail, see the paper published by Max Planck Institute for      *
 * Intelligent Systems on SIGGRAPH ASIA 2015.                                *
 *                                                                           *
 * We provide this software for research purposes only.                      *
 * The original SMPL model is available at http://smpl.is.tue.mpg.           *
 *                                                                           *
 * ========================================================================= */

//=============================================================================
//
//  UNIVERSAL VARIABLE DEFINITIONS
//
//=============================================================================

//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

#include "definition/def.h"

//===== EXTERNAL DECLARATIONS =================================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL DEFINITIONS ==================================================

/*
vertices_template.shape
(6890, 3)
face_indices.shape
(13776, 3)
weights.shape
(6890, 52)
shape_blend_shapes.shape
(6890, 3, 10)
pose_blend_shapes.shape
(6890, 3, 459)
joint_regressor.shape
(52, 6890)
kinematic_tree.shape
(2, 52)
*/

int64_t batch_size = 1;// 1
int64_t vertex_num = -1;// 6890;// 6890
int64_t joint_num = -1;//36;//52;// 24
int64_t shape_basis_dim = -1;// 10;// 10
int64_t pose_basis_dim = -1;// 400;//459;// 207
int64_t face_index_num = -1;// 13776;// 13776

//=============================================================================
} // namespace smpl
//=============================================================================

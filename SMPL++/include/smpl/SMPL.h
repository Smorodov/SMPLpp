#pragma once
#ifndef SMPL_H
#define SMPL_H
#include <string>

#define _CRT_SECURE_NO_DEPRECATE
// #define EIGEN_RUNTIME_NO_MALLOC // Define this symbol to enable runtime tests for allocations
// -------------------
// Eigen
// -------------------
// #include "Eigen/Eigen"
// -------------------
// zlib
// -------------------
#include "zlib/zlib.h"
#if defined(MSDOS) || defined(OS2) || defined(WIN32) || defined(__CYGWIN__)
#include <fcntl.h>
#include <io.h>
#define SET_BINARY_MODE(file) setmode(fileno(file), O_BINARY)
#else
#  define SET_BINARY_MODE(file)
#endif

#define CHECK_ERR(err, msg) { \
    if (err != Z_OK) { \
        fprintf(stderr, "%s error: %d\n", msg, err); \
        exit(1); \
    } \
}
// -------------------
// Numpy library
// -------------------
#include "cnpy.h"

//----------
#include <torch/torch.h>
//----------
#include "smpl/BlendShape.h"
#include "smpl/JointRegression.h"
#include "smpl/WorldTransformation.h"
#include "smpl/LinearBlendSkinning.h"

#define COUT_VAR(x) std::cout << #x"=" << x << std::endl;
#define COUT_ARR(x) std::cout << "---------"#x"---------" << std::endl;\
        COUT_ARR(x) std::cout << x << std::endl;\
        COUT_ARR(x) std::cout << "---------"#x"---------" << std::endl;
#define SHOW_IMG(x) cv::namedWindow(#x);cv::imshow(#x,x);cv::waitKey(20);

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS DEFINITIONS =====================================================

/**
 * DESCRIPTIONS:
 * 
 *      The final system to combine all modules and make them work properly.
 * 
 *      This class is system wrapper which dose real computation. The actually
 *      working process is consistent with the SMPL pipeline.
 * 
 * INHERITANCES:
 * 
 * 
 * ATTRIBUTES:
 * 
 *      - m__device: <private>
 *          Torch device to run the module, could be CPUs or GPUs.
 * 
 *      - m__modelPath: <private>
 *          Path to the JSON model file.
 * 
 *      - m__vertPath: <private>
 *          Path to store the mesh OBJ file.
 * 
 *      - m__faceIndices: <private>
 *          Vertex indices of each face, (13776, 3)
 * 
 *      - m__shapeBlendBasis: <private>
 *          Basis of the shape-dependent shape space,
 *          (6890, 3, 10).
 * 
 *      - m__poseBlendBasis: <private>
 *          Basis of the pose-dependent shape space, (6890, 3, 207).
 * 
 *      - m__templateRestShape: <private>
 *          Template shape in rest pose, (6890, 3).
 * 
 *      - m__jointRegressor: <private>
 *          Joint coefficients of each vertices for regressing them to joint
 *          locations, (24, 6890).
 * 
 *      - m__kinematicTree: <private>
 *          Hierarchy relation between joints, the root is at the belly button,
 *          (2, 24).
 * 
 *      - m__weights: <private>
 *          Weights for linear blend skinning, (6890, 24).
 * 
 *      - m__model: <private>
 *          JSON object represents.
 * 
 *      - m__blender: <private>
 *          A module to generate shape blend shape and pose blend shape
 *          by combining parameters thetas and betas with their own basis.
 * 
 *      - m__regressor: <private>
 *          A module to regress vertex position into joint location of the new
 *          shape with different pose deformation considered.
 * 
 *      - m__transformer: <private>
 *          A module to transform joints from T-pose's position into the ones 
 *          of new pose.
 * 
 *      - m__skinner: <private>
 *          A module to do linear blend skinning.
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Destructor.
 *      %
 *      - SMPL: <public>
 *          Default constructor.
 * 
 *      - SMPL: <public>
 *          Constructor to initialize model path, vertex path, and torch 
 *          device.
 * 
 *      - SMPL: <public>
 *          Copy constructor.
 * 
 *      - ~SMPL: <public>
 *          Destructor.
 *      %%
 * 
 *      %
 *          Operator
 *      %
 *      - operator=: <public>
 *          Assignment is used to copy a <SMPL> instantiation.
 *      %%
 * 
 *      %
 *          Setter and Getter
 *      %
 *      - setDevice: <public>
 *          Set the torch device.
 * 
 *      - setPath: <public>
 *          Set model path to the JSON model file.
 * 
 *      - getRestShape: <public>
 *          Get deformed shape in rest pose.
 * 
 *      - getFaceIndex: <public>
 *          Get vertex indices of each face.
 * 
 *      - getRestJoint: <public>
 *          Get joint locations of the deformed shape in rest pose.
 * 
 *      - getVertex: <public>
 *          Get vertex locations of the deformed mesh.
 *      %%
 * 
 *      %
 *          Modeling
 *      %
 *      - init: <public>
 *          Load model data stored as JSON file into current application.
 *          (Note: The loading will spend a long time because of a large
 *           JSON file.)
 * 
 *      - launch: <public>
 *          Run the model with a specific group of beta, theta, and 
 *          translation.
 * 
 *      - out: <public>
 *          Export the deformed mesh to OBJ file.
 *      %%
 * 
 */

class SMPL final
{
public:
private: // PIRVATE ATTRIBUTES

    torch::Device m__device;

    std::string m__modelPath;
    std::string m__vertPath;
   // nlohmann::json m__model;

    torch::Tensor m__faceIndices;
    torch::Tensor m__shapeBlendBasis;
    torch::Tensor m__poseBlendBasis;
    torch::Tensor m__templateRestShape;
    torch::Tensor m__jointRegressor;
    torch::Tensor m__kinematicTree;
    torch::Tensor m__weights;

    BlendShape m__blender;
    JointRegression m__regressor;
    WorldTransformation m__transformer;
    LinearBlendSkinning m__skinner;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Constructor and Destructor %%
    SMPL() noexcept(true);
    SMPL(std::string &modelPath, 
        std::string &vertPath, torch::Device &device) noexcept(false);
    SMPL(const SMPL& smpl) noexcept(false);
    ~SMPL() noexcept(true);

    // %% Operators %%
    SMPL &operator=(const SMPL& smpl) noexcept(false);

    // %% Setter and Getter %%
    void setDevice(const torch::Device &device) noexcept(false);
    void setModelPath(const std::string &modelPath) noexcept(false);
    void setVertPath(const std::string &vertexPath) noexcept(false);

    torch::Tensor getRestShape() noexcept(false);
    torch::Tensor getFaceIndex() noexcept(false);
    torch::Tensor getRestJoint() noexcept(false);
    torch::Tensor getVertex() noexcept(false);

    // %% Modeling %%
    void init() noexcept(false);
    void launch(
        torch::Tensor &beta,
        torch::Tensor &theta) noexcept(false);
    void out(int64_t index) noexcept(false);
    void getVandF(int64_t index, std::vector<float>& vx,
        std::vector<float>& vy,
        std::vector<float>& vz,
        std::vector<size_t>& f1,
        std::vector<size_t>& f2,
        std::vector<size_t>& f3
    ) noexcept(false);

    void getSkeleton(int64_t index,
        std::vector<int64_t>& l1,
        std::vector<int64_t>& l2,
        std::vector<float>& jx,
        std::vector<float>& jy,
        std::vector<float>& jz
    ) noexcept(false);

};

//=============================================================================
} // namespace smpl
//=============================================================================
#endif // SMPL_H
//=============================================================================

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
//  CLASS SMPL IMPLEMENTATIONS
//
//=============================================================================


//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include <fstream>
#include <filesystem>
//----------
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include "xtensor-io/xnpz.hpp"
//----------
#include "definition/def.h"
#include "toolbox/TorchEx.hpp"
#include "smpl/SMPL.h"
//----------
#include "iostream"

#define COUT_VAR(x) std::cout << #x"=" << x << std::endl;
#define COUT_ARR(x) std::cout << "---------"<< #x << "---------" << std::endl;\
        std::cout << x << std::endl;\
        std::cout << "--------------------" << std::endl;
#define SHOW_IMG(x) cv::namedWindow(#x);cv::imshow(#x,x);cv::waitKey(20);

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS IMPLEMENTATIONS =================================================

/**SMPL
 * 
 * Brief
 * ----------
 * 
 *      Default constructor.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
SMPL::SMPL() noexcept(true) :
    m__device(torch::kCPU),
    m__modelPath(),
    m__vertPath(),
    m__faceIndices(),
    m__shapeBlendBasis(),
    m__poseBlendBasis(),
    m__templateRestShape(),
    m__jointRegressor(),
    m__kinematicTree(),
    m__weights(),
    //m__model(),
    m__blender(),
    m__regressor(),
    m__transformer(),
    m__skinner()
{
}

/**SMPL (overload)
 * 
 * Brief
 * ----------
 * 
 *      Constructor to initialize model path.
 * 
 * Arguments
 * ----------
 * 
 *      @modelPath: - string -
 *          Model path to be specified.
 * 
 * Return
 * ----------
 * 
 * 
 */
SMPL::SMPL(std::string &modelPath, 
    std::string &vertPath, torch::Device &device) noexcept(false) :
    m__device(torch::kCPU),
    //m__model(),
    m__blender(),
    m__regressor(),
    m__transformer(),
    m__skinner()
{
    if (device.has_index()) {
        m__device = device;
    }
    else {
        throw smpl_error("SMPL", "Failed to fetch device index!");
    }

    std::filesystem::path path(modelPath);
    if (std::filesystem::exists(path)) {
        m__modelPath = modelPath;
        m__vertPath = vertPath;
    }
    else {
        throw smpl_error("SMPL", "Failed to initialize model path!");
    }
}

/**SMPL (overload)
 * 
 * Brief
 * ----------
 * 
 *      Copy constructor.
 * 
 * Arguments
 * ----------
 * 
 *      @smpl: - const SMPL& -
 *          The <LinearBlendSkinning> instantiation to copy with.
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
SMPL::SMPL(const SMPL& smpl) noexcept(false) :
    m__device(torch::kCPU)
{
    try {
        *this = smpl;
    }
    catch(std::exception &e) {
        throw;
    }
}

/**~SMPL
 * 
 * Brief
 * ----------
 * 
 *      Destructor
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
SMPL::~SMPL() noexcept(true)
{
}

/**operator=
 * 
 * Brief
 * ----------
 * 
 *      Assignment is used to copy a <SMPL> instantiation.
 * 
 * Arguments
 * ----------
 * 
 *      @smpl: - const SMPL& -
 *          The <SMPL> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 *      @this*: - SMPL & -
 *          Current instantiation.
 * 
 */
SMPL &SMPL::operator=(const SMPL& smpl) noexcept(false)
{
    //
    // hard copy
    //
    if (smpl.m__device.has_index()) {
        m__device = smpl.m__device;
    }
    else {
        throw smpl_error("SMPL", "Failed to fetch device index!");
    }

    std::filesystem::path path(smpl.m__modelPath);
    if (std::filesystem::exists(path)) {
        m__modelPath = smpl.m__modelPath;
    }
    else {
        throw smpl_error("SMPL", "Failed to copy model path!");
    }

    try {
        m__vertPath = smpl.m__vertPath;

        //m__model = smpl.m__model;
        m__blender = smpl.m__blender;
        m__regressor = smpl.m__regressor;
        m__transformer = smpl.m__transformer;
        m__skinner = smpl.m__skinner;
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // soft copy
    //
    if (smpl.m__faceIndices.sizes() ==
        torch::IntArrayRef({FACE_INDEX_NUM, 3})) {
        m__faceIndices = smpl.m__faceIndices.clone().to(m__device);
    }

    if (smpl.m__shapeBlendBasis.sizes() == 
        torch::IntArrayRef({vertex_num, 3, SHAPE_BASIS_DIM})) {
        m__shapeBlendBasis = smpl.m__shapeBlendBasis.clone().to(
            m__device);
    }

    if (smpl.m__poseBlendBasis.sizes() == 
        torch::IntArrayRef({vertex_num, 3, POSE_BASIS_DIM})) {
        m__poseBlendBasis = smpl.m__poseBlendBasis.clone().to(m__device);
    }

    if (smpl.m__jointRegressor.sizes() == 
        torch::IntArrayRef({joint_num, vertex_num})) {
        m__jointRegressor = smpl.m__jointRegressor.clone().to(m__device);
    }

    if (smpl.m__templateRestShape.sizes() ==
        torch::IntArrayRef({vertex_num, 3})) {
        m__templateRestShape = smpl.m__templateRestShape.clone().to(
            m__device);
    }

    if (smpl.m__kinematicTree.sizes() ==
        torch::IntArrayRef({2, joint_num})) {
        m__kinematicTree = smpl.m__kinematicTree.clone().to(m__device);
    }

    if (smpl.m__weights.sizes() ==
        torch::IntArrayRef({vertex_num, joint_num})) {
        m__weights = smpl.m__weights.clone().to(m__device);
    }

    return *this;
}

/**setDevice
 * 
 * Brief
 * ----------
 * 
 *      Set the torch device.
 * 
 * Arguments
 * ----------
 * 
 *      @device: - const Device & -
 *          The torch device to be used.
 * 
 * Return
 * ----------
 * 
 */
void SMPL::setDevice(const torch::Device &device) noexcept(false)
{
    if (device.has_index()) {
        m__device = device;
        m__blender.setDevice(device);
        m__regressor.setDevice(device);
        m__transformer.setDevice(device);
        m__skinner.setDevice(device);
    }
    else {
        throw smpl_error("SMPL", "Failed to fetch device index!");
    }

    return;
}

/**setModelPath
 * 
 * Brief
 * ----------
 * 
 *      Set model path to the JSON model file.
 * 
 * Arguments
 * ----------
 * 
 *      @modelPath: - string -
 *          Model path to be specified.
 * 
 * Return
 * ----------
 * 
 * 
 */
void SMPL::setModelPath(const std::string &modelPath) noexcept(false)
{
    std::filesystem::path path(modelPath);
    if (std::filesystem::exists(path)) {
        m__modelPath = modelPath;
    }
    else {
        throw smpl_error("SMPL", "Failed to initialize model path!");
    }

    return;
}

/**setVertPath
 * 
 * Brief
 * ----------
 * 
 *      Set path for exporting the mesh to OBJ file.
 * 
 * Arguments
 * ----------
 * 
 *      @vertexPath: - string -
 *          Vertex path to be specified.
 * 
 * Return
 * ----------
 * 
 * 
 */
void SMPL::setVertPath(const std::string &vertexPath) noexcept(false)
{
    m__vertPath = vertexPath;

    return;
}

/**getRestShape
 * 
 * Brief
 * ----------
 * 
 *      Get deformed shape in rest pose.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @restShape: - Tensor -
 *          Deformed shape in rest pose, (N, 6890, 3)
 * 
 */
torch::Tensor SMPL::getRestShape() noexcept(false)
{
    torch::Tensor restShape;
    
    try {
        restShape = m__regressor.getRestShape().clone().to(m__device);
    }
    catch(std::exception &e) {
        throw;
    }

    return restShape;
}

/**getFaceIndex
 * 
 * Brief
 * ----------
 * 
 *      Get vertex indices of each face.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @faceIndices: - Tensor -
 *          Vertex indices of each face (triangles), (13776, 3).
 * 
 */
torch::Tensor SMPL::getFaceIndex() noexcept(false)
{
    torch::Tensor faceIndices;
    if (m__faceIndices.sizes() !=
        torch::IntArrayRef(
            {FACE_INDEX_NUM, 3})) {
        faceIndices = m__faceIndices.clone().to(m__device);
    }
    else {
        throw smpl_error("SMPL", "Failed to get face indices!");
    }

    return faceIndices;
}

/**getRestJoint
 * 
 * Brief
 * ----------
 * 
 *      Get joint locations of the deformed shape in rest pose.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @joints: - Tensor -
 *          Joint locations of the deformed mesh in rest pose, (N, 24, 3).
 * 
 */
torch::Tensor SMPL::getRestJoint() noexcept(false)
{
    torch::Tensor joints;
    
    try {
        joints = m__regressor.getJoint().clone().to(m__device);
    }
    catch (std::exception &e) {
        throw;
    }

    return joints;
}

/**getVertex
 * 
 * Brief
 * ----------
 * 
 *      Get vertex locations of the deformed mesh.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @vertices: - Tensor -
 *          Vertex locations of the deformed mesh, (N, 6890, 3).
 * 
 */
torch::Tensor SMPL::getVertex() noexcept(false)
{
    torch::Tensor vertices;

    try {
        vertices = m__skinner.getVertex().clone().to(m__device);
    }
    catch(std::exception &e) {
        throw;
    }

    return vertices;
}

/**init
 * 
 * Brief
 * ----------
 * 
 *          Load model data stored as JSON file into current application.
 *          (Note: The loading will spend a long time because of a large
 *           JSON file.)
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
void SMPL::init() noexcept(false)
{          
    std::cout << "---------------------------"<< std::endl;
    std::cout << "Loding SMPL model file: " << m__modelPath << std::endl;
    std::cout << "---------------------------" << std::endl;
    auto npz_map = xt::load_npz(m__modelPath);
        //
        // data loading
        //
        // face indices
        xt::xarray<uint32_t> faceIndices=npz_map["face_indices"].cast<uint32_t>();                
        
        face_index_num = faceIndices.shape()[0];
        COUT_VAR(face_index_num);

        m__faceIndices = torch::from_blob(faceIndices.data(),
            {FACE_INDEX_NUM, 3}, torch::kInt32).clone().to(
                m__device);

        xt::xarray<float> templateRestShape = npz_map["vertices_template"].cast<double>();
        vertex_num = templateRestShape.shape()[0];
        COUT_VAR(vertex_num);

        // blender
        xt::xarray<float> shapeBlendBasis = npz_map["shape_blend_shapes"].cast<double>();
        xt::xarray<float> poseBlendBasis = npz_map["pose_blend_shapes"].cast<double>();

        shape_basis_dim = shapeBlendBasis.shape()[2];
        COUT_VAR(shape_basis_dim);

        pose_basis_dim = poseBlendBasis.shape()[2];
        COUT_VAR(pose_basis_dim);
        
        m__shapeBlendBasis = torch::from_blob(shapeBlendBasis.data(),
            {vertex_num, 3, shape_basis_dim},torch::kF32).to(m__device);// (6890, 3, 10)
        
        
        m__poseBlendBasis = torch::from_blob(poseBlendBasis.data(),
            {vertex_num, 3, pose_basis_dim}, torch::kF32).to(m__device);// (6890, 3, 207)

        // regressor
        xt::xarray<float> jointRegressor = npz_map["joint_regressor"].cast<double>();

        joint_num = jointRegressor.shape()[0];
        COUT_VAR(joint_num);

        
        m__templateRestShape = torch::from_blob(templateRestShape.data(),
            {vertex_num, 3}).to(m__device);// (6890, 3)
      
        m__jointRegressor = torch::from_blob(jointRegressor.data(),
            {joint_num, vertex_num}).to(m__device);// (24, 6890)

        // transformer
        xt::xarray<int64_t> kinematicTree = npz_map["kinematic_tree"].cast<int64_t>();                
        m__kinematicTree = torch::from_blob(kinematicTree.data(),
            { 2, joint_num },torch::kInt64).to(m__device);// (2, 24)                
        // skinner
        xt::xarray<float> weights = npz_map["weights"].cast<double>();
       // xt::from_json(m__model["weights"], weights);
        m__weights = torch::from_blob(weights.data(),
            {vertex_num, joint_num}).to(m__device);// (6890, 24)
        std::cout << "---------------------------" << std::endl;
        std::cout << " Successfully loaded " << std::endl;
        std::cout << "---------------------------" << std::endl;

    return;
}

/**launch
 * 
 * Brief
 * ----------
 * 
 *          Run the model with a specific group of beta, theta, and 
 *          translation.
 * 
 * Arguments
 * ----------
 * 
 *      @beta: - Tensor -
 *          Batch of shape coefficient vectors, (N, 10).
 * 
 *      @theta: - Tensor -
 *          Batch of pose in axis-angle representations, (N, 24, 3).
 * 
 *      @translation: - Tensor -
 *          Batch of global translation vectors, (N, 3).
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
void SMPL::launch(
    torch::Tensor &beta, 
    torch::Tensor &theta) noexcept(false)
{
    try {
        //
        // blend shapes
        //
        m__blender.setBeta(beta);
        m__blender.setTheta(theta);
        m__blender.setShapeBlendBasis(m__shapeBlendBasis);
        m__blender.setPoseBlendBasis(m__poseBlendBasis);

        m__blender.blend();

        torch::Tensor shapeBlendShape = m__blender.getShapeBlendShape();
        torch::Tensor poseBlendShape = m__blender.getPoseBlendShape();
        torch::Tensor poseRotation = m__blender.getPoseRotation();

        //
        // regress joints
        //
        m__regressor.setTemplateRestShape(m__templateRestShape);
        m__regressor.setJointRegressor(m__jointRegressor);
        m__regressor.setShapeBlendShape(shapeBlendShape);
        m__regressor.setPoseBlendShape(poseBlendShape);

        m__regressor.regress();

        torch::Tensor restShape = m__regressor.getRestShape();
        torch::Tensor joints = m__regressor.getJoint();

        //
        // transform
        //
        m__transformer.setKinematicTree(m__kinematicTree);
        m__transformer.setJoint(joints);
        m__transformer.setPoseRotation(poseRotation);

        m__transformer.transform();

        torch::Tensor transformation = m__transformer.getTransformation();

        //
        // skinning
        //
        m__skinner.setWeight(m__weights);
        m__skinner.setRestShape(restShape);
        m__skinner.setTransformation(transformation);

        m__skinner.skinning();
    }
    catch(std::exception &e) {
        throw;
    }

    return;
}

/**out
 * 
 * Brief
 * ----------
 * 
 *      Export the deformed mesh to OBJ file.
 * 
 * Arguments
 * ----------
 * 
 *      @index: - size_t -
 *          A mesh in the batch to be exported.
 * 
 * Return
 * ----------
 * 
 * 
 */
void SMPL::out(int64_t index) noexcept(false)
{
    torch::Tensor vertices = 
        m__skinner.getVertex().clone().to(m__device);// (N, 6890, 3)

    if (vertices.sizes() ==
            torch::IntArrayRef(
                {batch_size, vertex_num, 3})
        && m__faceIndices.sizes() ==
            torch::IntArrayRef(
                {FACE_INDEX_NUM, 3})
        ) {
        std::ofstream file(m__vertPath);

        torch::Tensor slice_ = TorchEx::indexing(vertices,
            torch::IntList({index}));// (6890, 3)
        xt::xarray<float> slice = xt::adapt(
            (float *)slice_.to(torch::kCPU).data_ptr(),
            xt::xarray<float>::shape_type({(const size_t)vertex_num, 3})
        );
        
        xt::xarray<int32_t> faceIndices;
        faceIndices = xt::adapt(
            (int32_t *)m__faceIndices.to(torch::kCPU).data_ptr(),
            xt::xarray<int32_t>::shape_type(
                {(const size_t)FACE_INDEX_NUM, 3})
        );

        for (int64_t i = 0; i < vertex_num; i++) {
            file << 'v' << ' '
                << slice(i, 0) << ' '
                << slice(i, 1) << ' ' 
                << slice(i, 2) << '\n';
        }

        for (int64_t i = 0; i < FACE_INDEX_NUM; i++) {
            file << 'f' << ' '
                << faceIndices(i, 0) << ' '
                << faceIndices(i, 1) << ' '
                << faceIndices(i, 2) << '\n';
        }
    }
    else {
        throw smpl_error("SMPL", "Cannot export the deformed mesh!");
    }

    return;
}




void SMPL::getVandF(int64_t index,
    std::vector<float>& vx,
    std::vector<float>& vy,
    std::vector<float>& vz,
    std::vector<size_t>& f1,
    std::vector<size_t>& f2,
    std::vector<size_t>& f3
    ) noexcept(false)
{
    vx.clear();
    vy.clear();
    vz.clear();
    f1.clear();
    f2.clear();
    f3.clear();

    torch::Tensor vertices =
        m__skinner.getVertex().clone().to(m__device);// (N, 6890, 3)

    if (vertices.sizes() ==
        torch::IntArrayRef(
            { batch_size, vertex_num, 3 })
        && m__faceIndices.sizes() ==
        torch::IntArrayRef(
            { FACE_INDEX_NUM, 3 })
        )
    {
 
        torch::Tensor slice_ = TorchEx::indexing(vertices,
            torch::IntList({ index }));// (6890, 3)
        xt::xarray<float> slice = xt::adapt(
            (float*)slice_.to(torch::kCPU).data_ptr(),
            xt::xarray<float>::shape_type({ (const size_t)vertex_num, 3 })
        );

        xt::xarray<int32_t> faceIndices;
        faceIndices = xt::adapt(
            (int32_t*)m__faceIndices.to(torch::kCPU).data_ptr(),
            xt::xarray<int32_t>::shape_type(
                { (const size_t)FACE_INDEX_NUM, 3 })
        );

        for (int64_t i = 0; i < vertex_num; i++)
        {
            vx.push_back(slice(i, 0));
            vy.push_back(slice(i, 1));
            vz.push_back(slice(i, 2));
        }

        for (int64_t i = 0; i < FACE_INDEX_NUM; i++)
        {
            f1.push_back(faceIndices(i, 0));
            f2.push_back(faceIndices(i, 1));
            f3.push_back(faceIndices(i, 2));
        }
    }
    else
    {
        throw smpl_error("SMPL", "Cannot export the deformed mesh!");
    }

    return;
}

//=============================================================================
} // namespace smpl
//=============================================================================

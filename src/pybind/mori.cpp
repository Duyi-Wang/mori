// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "src/pybind/mori.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mori/application/application.hpp"
#include "mori/io/io.hpp"
#include "mori/ops/ops.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/pybind/dlpack_utils.hpp"

/* ---------------------------------------------------------------------------------------------- */
/*                                            Ops APIs                                            */
/* ---------------------------------------------------------------------------------------------- */
namespace {

std::tuple<py::capsule, std::optional<py::capsule>, std::optional<py::capsule>, py::capsule,
           py::capsule>
LaunchDispatch(mori::moe::EpDispatchCombineHandle& handle, int kernelType, py::capsule input,
               const std::optional<py::capsule>& weights, const std::optional<py::capsule>& scales,
               py::capsule topkIds, int blockNum, int warpPerBlock, uint64_t stream_ptr) {
  DLTensor* inputTensor = mori::GetDLTensor(input);
  DLTensor* topkIdsTensor = mori::GetDLTensor(topkIds);

  assert(mori::IsDLTensorContiguous(inputTensor) && mori::IsDLTensorContiguous(topkIdsTensor));

  float* weightPtr = nullptr;
  if (weights.has_value()) {
    DLTensor* weightsTensor = mori::GetDLTensor(weights.value());
    assert(mori::IsDLTensorContiguous(weightsTensor) &&
           mori::GetDLTensorElementSize(weightsTensor) == sizeof(float));
    weightPtr = static_cast<float*>(weightsTensor->data);
  }

  uint8_t* scalePtr = nullptr;
  DLTensor* scalesTensor = nullptr;
  if (scales.has_value() && handle.config.scaleDim > 0) {
    scalesTensor = mori::GetDLTensor(scales.value());
    assert(mori::IsDLTensorContiguous(scalesTensor) &&
           mori::GetDLTensorElementSize(scalesTensor) == handle.config.scaleTypeSize);
    scalePtr = static_cast<uint8_t*>(scalesTensor->data);
  }

  handle.PrepareInference(
      mori::DLPackDtypeToHipDataType(inputTensor->dtype), inputTensor->data, nullptr, weightPtr,
      scalePtr, static_cast<mori::moe::index_t*>(topkIdsTensor->data), inputTensor->shape[0]);

  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
  handle.LaunchDispatch((mori::moe::KernelType)kernelType, blockNum, warpPerBlock, stream);

  py::capsule out =
      mori::CreateDLPackCapsule(handle.shmemDispatchOutTokMemObj->Get(),
                                {handle.config.MaxNumTokensToRecv(), handle.config.hiddenDim},
                                inputTensor->dtype, inputTensor->device.device_id);

  std::optional<py::capsule> outWeights{std::nullopt};
  if (weights.has_value()) {
    DLDataType float32Type = mori::GetDLPackDataType<float>();
    outWeights = mori::CreateDLPackCapsule(
        handle.shmemDispatchOutWeightsMemObj->Get(),
        {handle.config.MaxNumTokensToRecv(), handle.config.numExpertPerToken}, float32Type,
        inputTensor->device.device_id);
  }

  std::optional<py::capsule> outScales{std::nullopt};
  if (scales.has_value() && handle.config.scaleDim > 0) {
    outScales =
        mori::CreateDLPackCapsule(handle.shmemOutScalesMemObj->Get(),
                                  {handle.config.MaxNumTokensToRecv(), handle.config.scaleDim},
                                  scalesTensor->dtype, inputTensor->device.device_id);
  }

  DLDataType indexType = mori::GetDLPackDataType<mori::moe::index_t>();
  py::capsule outIndices = mori::CreateDLPackCapsule(
      handle.shmemOutIndicesMemObj->Get(),
      {handle.config.MaxNumTokensToRecv(), handle.config.numExpertPerToken}, indexType,
      inputTensor->device.device_id);

  py::capsule totalRecvTokenNum = mori::CreateDLPackCapsule(
      handle.totalRecvTokenNum, {1}, indexType, inputTensor->device.device_id);

  return {out, outWeights, outScales, outIndices, totalRecvTokenNum};
}

std::tuple<py::capsule, std::optional<py::capsule>> LaunchCombine(
    mori::moe::EpDispatchCombineHandle& handle, int kernelType, py::capsule input,
    const std::optional<py::capsule>& weights, py::capsule topkIds, int blockNum, int warpPerBlock,
    uint64_t stream_ptr) {
  DLTensor* inputTensor = mori::GetDLTensor(input);
  DLTensor* topkIdsTensor = mori::GetDLTensor(topkIds);

  assert(mori::IsDLTensorContiguous(inputTensor) && mori::IsDLTensorContiguous(topkIdsTensor));

  float* weightsPtr = nullptr;
  DLTensor* weightsTensor = nullptr;
  if (weights.has_value()) {
    weightsTensor = mori::GetDLTensor(weights.value());
    if (weightsTensor->shape[0] != 0) {
      assert(mori::IsDLTensorContiguous(weightsTensor));
      weightsPtr = static_cast<float*>(weightsTensor->data);
    }
  }

  handle.PrepareInference(
      mori::DLPackDtypeToHipDataType(inputTensor->dtype), inputTensor->data, nullptr, weightsPtr,
      static_cast<mori::moe::index_t*>(topkIdsTensor->data), handle.curRankNumToken);

  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
  handle.LaunchCombine((mori::moe::KernelType)kernelType, blockNum, warpPerBlock, stream);

  py::capsule out =
      mori::CreateDLPackCapsule(handle.shmemCombineOutTokMemObj->Get(),
                                {handle.config.maxNumInpTokenPerRank, handle.config.hiddenDim},
                                inputTensor->dtype, inputTensor->device.device_id);

  std::optional<py::capsule> outWeights{std::nullopt};
  if (weightsPtr && weightsTensor != nullptr) {
    outWeights = mori::CreateDLPackCapsule(
        handle.shmemCombineOutWeightsMemObj->Get(),
        {handle.config.maxNumInpTokenPerRank, handle.config.numExpertPerToken},
        weightsTensor->dtype, inputTensor->device.device_id);
  }

  return {out, outWeights};
}

void LaunchReset(mori::moe::EpDispatchCombineHandle& handle, uint64_t stream_ptr) {
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
  handle.LaunchReset(stream);
}

py::capsule GetDispatchSrcTokenId(mori::moe::EpDispatchCombineHandle& handle) {
  DLDataType indexType = mori::GetDLPackDataType<mori::moe::index_t>();
  py::capsule tensor = mori::CreateDLPackCapsule(
      handle.dispTokIdToSrcTokIdMemObj->template GetAs<mori::moe::index_t*>(),
      {*handle.totalRecvTokenNum}, indexType, 0);
  return tensor;
}

py::capsule GetDispatchSenderTokenIdxMap(mori::moe::EpDispatchCombineHandle& handle) {
  DLDataType indexType = mori::GetDLPackDataType<mori::moe::index_t>();
  py::capsule tensor = mori::CreateDLPackCapsule(
      handle.dispSenderIdxMap, {handle.curRankNumToken * handle.config.numExpertPerToken},
      indexType, 0);
  return tensor;
}

py::capsule GetDispatchReceiverTokenIdxMap(mori::moe::EpDispatchCombineHandle& handle) {
  DLDataType indexType = mori::GetDLPackDataType<mori::moe::index_t>();
  py::capsule tensor = mori::CreateDLPackCapsule(handle.dispReceiverIdxMap,
                                                 {*handle.localPeTokenCounter}, indexType, 0);
  return tensor;
}

py::capsule GetRegisteredCombineInputBuffer(mori::moe::EpDispatchCombineHandle& handle,
                                            mori::DLPackDtype dtype) {
  DLDataType dlType = mori::PythonDtypeToDLPack(dtype);
  py::capsule out = mori::CreateDLPackCapsule(
      handle.shmemCombineInpTokMemObj->Get(),
      {handle.config.MaxNumTokensToRecv(), handle.config.hiddenDim}, dlType, 0);
  return out;
}

void DeclareEpDispatchCombineHandle(pybind11::module& m) {
  std::string className = std::string("EpDispatchCombineHandle");
  pybind11::class_<mori::moe::EpDispatchCombineHandle>(m, className.c_str())
      .def(pybind11::init<mori::moe::EpDispatchCombineConfig>(),
           py::arg("config") = mori::moe::EpDispatchCombineConfig{});

  std::string funcName = std::string("launch_dispatch");
  m.def(funcName.c_str(), &LaunchDispatch);

  funcName = std::string("launch_combine");
  m.def(funcName.c_str(), &LaunchCombine);

  funcName = std::string("launch_reset");
  m.def(funcName.c_str(), &LaunchReset);

  funcName = std::string("get_cur_rank_num_token");
  m.def(funcName.c_str(), &mori::moe::EpDispatchCombineHandle::GetCurRankNumToken);

  funcName = std::string("get_dispatch_src_token_pos");
  m.def(funcName.c_str(), &GetDispatchSrcTokenId);

  funcName = std::string("get_dispatch_sender_token_idx_map");
  m.def(funcName.c_str(), &GetDispatchSenderTokenIdxMap);

  funcName = std::string("get_dispatch_receiver_token_idx_map");
  m.def(funcName.c_str(), &GetDispatchReceiverTokenIdxMap);

  funcName = std::string("get_registered_combine_input_buffer");
  m.def(funcName.c_str(), &GetRegisteredCombineInputBuffer);
}

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                           Shmem APIs                                           */
/* ---------------------------------------------------------------------------------------------- */
namespace {
int64_t ShmemTorchProcessGroupInit(const std::string& groupName) {
  return mori::shmem::ShmemTorchProcessGroupInit(groupName);
}

int64_t ShmemFinalize() { return mori::shmem::ShmemFinalize(); }

int64_t ShmemMyPe() { return mori::shmem::ShmemMyPe(); }

int64_t ShmemNPes() { return mori::shmem::ShmemNPes(); }

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                             IO APIs                                            */
/* ---------------------------------------------------------------------------------------------- */
namespace {}

namespace mori {

void RegisterMoriOps(py::module_& m) {
  pybind11::enum_<mori::DLPackDtype>(m, "DLPackDtype")
      .value("Float32", mori::DLPackDtype::kFloat32)
      .value("Float16", mori::DLPackDtype::kFloat16)
      .value("BFloat16", mori::DLPackDtype::kBFloat16)
      .value("Int32", mori::DLPackDtype::kInt32)
      .value("Int64", mori::DLPackDtype::kInt64)
      .value("UInt32", mori::DLPackDtype::kUInt32)
      .value("UInt64", mori::DLPackDtype::kUInt64)
      .value("Float8_e4m3fnuz", mori::DLPackDtype::kFloat8_e4m3fnuz)
      .export_values();

  pybind11::enum_<mori::moe::KernelType>(m, "EpDispatchCombineKernelType")
      .value("IntraNode", mori::moe::KernelType::IntraNode)
      .value("InterNode", mori::moe::KernelType::InterNode)
      .value("InterNodeV1", mori::moe::KernelType::InterNodeV1)
      .value("InterNodeV1LL", mori::moe::KernelType::InterNodeV1LL)
      .export_values();

  pybind11::class_<mori::moe::EpDispatchCombineConfig>(m, "EpDispatchCombineConfig")
      .def(pybind11::init<int, int, int, int, int, int, int, int, int, int, int, bool, int, int>(),
           py::arg("rank") = 0, py::arg("world_size") = 0, py::arg("hidden_dim") = 0,
           py::arg("scale_dim") = 0, py::arg("scale_type_size") = 0,
           py::arg("max_token_type_size") = 0, py::arg("max_num_inp_token_per_rank") = 0,
           py::arg("num_experts_per_rank") = 0, py::arg("num_experts_per_token") = 0,
           py::arg("warp_num_per_block") = 0, py::arg("block_num") = 0,
           py::arg("use_external_inp_buf") = true, py::arg("gpu_per_node") = 8,
           py::arg("rdma_block_num") = 0)
      .def_readwrite("rank", &mori::moe::EpDispatchCombineConfig::rank)
      .def_readwrite("world_size", &mori::moe::EpDispatchCombineConfig::worldSize)
      .def_readwrite("hidden_dim", &mori::moe::EpDispatchCombineConfig::hiddenDim)
      .def_readwrite("scale_dim", &mori::moe::EpDispatchCombineConfig::scaleDim)
      .def_readwrite("scale_type_size", &mori::moe::EpDispatchCombineConfig::scaleTypeSize)
      .def_readwrite("max_token_type_size", &mori::moe::EpDispatchCombineConfig::maxTokenTypeSize)
      .def_readwrite("max_num_inp_token_per_rank",
                     &mori::moe::EpDispatchCombineConfig::maxNumInpTokenPerRank)
      .def_readwrite("num_experts_per_rank", &mori::moe::EpDispatchCombineConfig::numExpertPerRank)
      .def_readwrite("num_experts_per_token",
                     &mori::moe::EpDispatchCombineConfig::numExpertPerToken)
      .def_readwrite("warp_num_per_block", &mori::moe::EpDispatchCombineConfig::warpNumPerBlock)
      .def_readwrite("block_num", &mori::moe::EpDispatchCombineConfig::blockNum)
      .def_readwrite("gpu_per_node", &mori::moe::EpDispatchCombineConfig::gpuPerNode)
      .def_readwrite("rdma_block_num", &mori::moe::EpDispatchCombineConfig::rdmaBlockNum);

  DeclareEpDispatchCombineHandle(m);
}

void RegisterMoriShmem(py::module_& m) {
  m.def("shmem_torch_process_group_init", &ShmemTorchProcessGroupInit);
  m.def("shmem_finalize", &ShmemFinalize);
  m.def("shmem_mype", &ShmemMyPe);
  m.def("shmem_npes", &ShmemNPes);
}

void RegisterMoriIo(pybind11::module_& m) {
  m.def("set_log_level", &mori::io::SetLogLevel);

  py::enum_<mori::io::BackendType>(m, "BackendType")
      .value("Unknown", mori::io::BackendType::Unknown)
      .value("XGMI", mori::io::BackendType::XGMI)
      .value("RDMA", mori::io::BackendType::RDMA)
      .value("TCP", mori::io::BackendType::TCP)
      .export_values();

  py::enum_<mori::io::MemoryLocationType>(m, "MemoryLocationType")
      .value("Unknown", mori::io::MemoryLocationType::Unknown)
      .value("CPU", mori::io::MemoryLocationType::CPU)
      .value("GPU", mori::io::MemoryLocationType::GPU)
      .export_values();

  py::enum_<mori::io::StatusCode>(m, "StatusCode")
      .value("SUCCESS", mori::io::StatusCode::SUCCESS)
      .value("INIT", mori::io::StatusCode::INIT)
      .value("IN_PROGRESS", mori::io::StatusCode::IN_PROGRESS)
      .value("ERR_INVALID_ARGS", mori::io::StatusCode::ERR_INVALID_ARGS)
      .value("ERR_NOT_FOUND", mori::io::StatusCode::ERR_NOT_FOUND)
      .value("ERR_RDMA_OP", mori::io::StatusCode::ERR_RDMA_OP)
      .value("ERR_BAD_STATE", mori::io::StatusCode::ERR_BAD_STATE)
      .export_values();

  py::enum_<mori::io::PollCqMode>(m, "PollCqMode")
      .value("POLLING", mori::io::PollCqMode::POLLING)
      .value("EVENT", mori::io::PollCqMode::EVENT);

  py::class_<mori::io::BackendConfig>(m, "BackendConfig");

  py::class_<mori::io::RdmaBackendConfig, mori::io::BackendConfig>(m, "RdmaBackendConfig")
      .def(py::init<int, int, int, mori::io::PollCqMode>(), py::arg("qp_per_transfer") = 1,
           py::arg("post_batch_size") = -1, py::arg("num_worker_threads") = -1,
           py::arg("poll_cq_mode") = mori::io::PollCqMode::POLLING)
      .def_readwrite("qp_per_transfer", &mori::io::RdmaBackendConfig::qpPerTransfer)
      .def_readwrite("post_batch_size", &mori::io::RdmaBackendConfig::postBatchSize)
      .def_readwrite("num_worker_threads", &mori::io::RdmaBackendConfig::numWorkerThreads)
      .def_readwrite("poll_cq_mode", &mori::io::RdmaBackendConfig::pollCqMode);

  py::class_<mori::io::IOEngineConfig>(m, "IOEngineConfig")
      .def(py::init<std::string, uint16_t>(), py::arg("host") = "", py::arg("port") = 0)
      .def_readwrite("host", &mori::io::IOEngineConfig::host)
      .def_readwrite("port", &mori::io::IOEngineConfig::port);

  py::class_<mori::io::TransferStatus>(m, "TransferStatus")
      .def(py::init<>())
      .def("Code", &mori::io::TransferStatus::Code)
      .def("Message", &mori::io::TransferStatus::Message)
      .def("Init", &mori::io::TransferStatus::Init)
      .def("InProgress", &mori::io::TransferStatus::InProgress)
      .def("Succeeded", &mori::io::TransferStatus::Succeeded)
      .def("Failed", &mori::io::TransferStatus::Failed)
      .def("SetCode", &mori::io::TransferStatus::SetCode)
      .def("SetMessage", &mori::io::TransferStatus::SetMessage)
      .def("Wait", &mori::io ::TransferStatus::Wait);

  py::class_<mori::io::EngineDesc>(m, "EngineDesc")
      .def_readonly("key", &mori::io::EngineDesc::key)
      .def_readonly("hostname", &mori::io::EngineDesc::hostname)
      .def_readonly("host", &mori::io::EngineDesc::host)
      .def_readonly("port", &mori::io::EngineDesc::port)
      .def(pybind11::self == pybind11::self)
      .def("pack",
           [](const mori::io::EngineDesc& d) {
             msgpack::sbuffer buf;
             msgpack::pack(buf, d);
             return py::bytes(buf.data(), buf.size());
           })
      .def_static("unpack", [](const py::bytes& b) {
        Py_ssize_t len = PyBytes_Size(b.ptr());
        const char* data = PyBytes_AsString(b.ptr());
        auto out = msgpack::unpack(data, len);
        return out.get().as<mori::io::EngineDesc>();
      });

  py::class_<mori::io::MemoryDesc>(m, "MemoryDesc")
      .def(py::init<>())
      .def_readonly("engine_key", &mori::io::MemoryDesc::engineKey)
      .def_readonly("id", &mori::io::MemoryDesc::id)
      .def_readonly("device_id", &mori::io::MemoryDesc::deviceId)
      .def_property_readonly("data",
                             [](const mori::io::MemoryDesc& desc) -> uintptr_t {
                               return reinterpret_cast<uintptr_t>(desc.data);
                             })
      .def_readonly("size", &mori::io::MemoryDesc::size)
      .def_readonly("loc", &mori::io::MemoryDesc::loc)
      .def(pybind11::self == pybind11::self)
      .def("pack",
           [](const mori::io::MemoryDesc& d) {
             msgpack::sbuffer buf;
             msgpack::pack(buf, d);
             return py::bytes(buf.data(), buf.size());
           })
      .def_static("unpack", [](const py::bytes& b) {
        Py_ssize_t len = PyBytes_Size(b.ptr());
        const char* data = PyBytes_AsString(b.ptr());
        auto out = msgpack::unpack(data, len);
        return out.get().as<mori::io::MemoryDesc>();
      });

  py::class_<mori::io::IOEngineSession>(m, "IOEngineSession")
      .def("AllocateTransferUniqueId", &mori::io ::IOEngineSession::AllocateTransferUniqueId)
      .def("Read", &mori::io ::IOEngineSession::Read)
      .def("BatchRead", &mori::io ::IOEngineSession::BatchRead)
      .def("Write", &mori::io ::IOEngineSession::Write)
      .def("BatchWrite", &mori::io ::IOEngineSession::BatchWrite)
      .def("Alive", &mori::io ::IOEngineSession::Alive);

  py::class_<mori::io::IOEngine>(m, "IOEngine")
      .def(py::init<const mori::io::EngineKey&, const mori::io::IOEngineConfig&>())
      .def("GetEngineDesc", &mori::io ::IOEngine::GetEngineDesc)
      .def("CreateBackend", &mori::io::IOEngine::CreateBackend)
      .def("RemoveBackend", &mori::io ::IOEngine::RemoveBackend)
      .def("RegisterRemoteEngine", &mori::io ::IOEngine::RegisterRemoteEngine)
      .def("DeregisterRemoteEngine", &mori::io ::IOEngine::DeregisterRemoteEngine)
      .def("RegisterMemory", &mori::io ::IOEngine::RegisterMemory)
      .def("DeregisterMemory", &mori::io ::IOEngine::DeregisterMemory)
      .def("AllocateTransferUniqueId", &mori::io ::IOEngine::AllocateTransferUniqueId)
      .def("Read", &mori::io ::IOEngine::Read)
      .def("BatchRead", &mori::io ::IOEngine::BatchRead)
      .def("Write", &mori::io ::IOEngine::Write)
      .def("BatchWrite", &mori::io ::IOEngine::BatchWrite)
      .def("CreateSession", &mori::io::IOEngine::CreateSession)
      .def("PopInboundTransferStatus", &mori::io::IOEngine::PopInboundTransferStatus);
}

}  // namespace mori

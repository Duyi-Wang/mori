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
#pragma once

#include <dlpack/dlpack.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/library_types.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mori {

// DLPack data type enum
enum class DLPackDtype : uint8_t {
  kFloat32 = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
  kInt32 = 3,
  kInt64 = 4,
  kUInt32 = 5,
  kUInt64 = 6,
  kFloat8_e4m3fnuz = 7,
};

// Convert C++ type to DLPack dtype code
template <typename T>
inline DLDataType GetDLPackDataType() {
  DLDataType dtype;
  if constexpr (std::is_same_v<T, float>) {
    dtype.code = kDLFloat;
    dtype.bits = 32;
    dtype.lanes = 1;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    dtype.code = kDLUInt;
    dtype.bits = 32;
    dtype.lanes = 1;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    dtype.code = kDLInt;
    dtype.bits = 32;
    dtype.lanes = 1;
  } else if constexpr (std::is_same_v<T, size_t> || std::is_same_v<T, uint64_t>) {
    dtype.code = kDLUInt;
    dtype.bits = 64;
    dtype.lanes = 1;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    dtype.code = kDLInt;
    dtype.bits = 64;
    dtype.lanes = 1;
  } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
    dtype.code = kDLBfloat;
    dtype.bits = 16;
    dtype.lanes = 1;
  } else if constexpr (std::is_same_v<T, __hip_fp8_e4m3_fnuz>) {
    dtype.code = kDLFloat8_e4m3fnuz;
    dtype.bits = 8;
    dtype.lanes = 1;
  } else {
    static_assert(sizeof(T) == 0, "Unsupported data type");
  }
  return dtype;
}

// Convert DLPack dtype to HIP data type
inline hipDataType DLPackDtypeToHipDataType(const DLDataType& dtype) {
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    return HIP_R_32F;
  } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
    return HIP_R_16BF;
  } else if (dtype.code == kDLFloat && dtype.bits == 8) {
    return HIP_R_8F_E4M3_FNUZ;
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    return HIP_R_16F;
  } else {
    throw std::runtime_error("Unsupported DLPack dtype for HIP");
  }
}

// Convert Python DLPack dtype enum to DLDataType
inline DLDataType PythonDtypeToDLPack(DLPackDtype dtype) {
  DLDataType dl_dtype;
  dl_dtype.lanes = 1;

  switch (dtype) {
    case DLPackDtype::kFloat32:
      dl_dtype.code = kDLFloat;
      dl_dtype.bits = 32;
      break;
    case DLPackDtype::kFloat16:
      dl_dtype.code = kDLFloat;
      dl_dtype.bits = 16;
      break;
    case DLPackDtype::kBFloat16:
      dl_dtype.code = kDLBfloat;
      dl_dtype.bits = 16;
      break;
    case DLPackDtype::kInt32:
      dl_dtype.code = kDLInt;
      dl_dtype.bits = 32;
      break;
    case DLPackDtype::kInt64:
      dl_dtype.code = kDLInt;
      dl_dtype.bits = 64;
      break;
    case DLPackDtype::kUInt32:
      dl_dtype.code = kDLUInt;
      dl_dtype.bits = 32;
      break;
    case DLPackDtype::kUInt64:
      dl_dtype.code = kDLUInt;
      dl_dtype.bits = 64;
      break;
    case DLPackDtype::kFloat8_e4m3fnuz:
      dl_dtype.code = kDLFloat;
      dl_dtype.bits = 8;
      break;
    default:
      throw std::runtime_error("Unsupported DLPackDtype");
  }
  return dl_dtype;
}

// Extract DLManagedTensor from Python object
inline DLManagedTensor* FromPyCapsule(py::capsule capsule) {
  PyObject* obj = capsule.ptr();
  DLManagedTensor* managed = nullptr;

  if (PyCapsule_IsValid(obj, "dltensor")) {
    managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(obj, "dltensor"));
  } else if (PyCapsule_IsValid(obj, "used_dltensor")) {
    managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(obj, "used_dltensor"));
  }

  if (!managed) {
    throw std::runtime_error("Invalid DLPack capsule: expected 'dltensor' or 'used_dltensor'");
  }

  return managed;
}

// Create a DLManagedTensor from raw data
// Note: This creates a non-owning view of the data
struct DLManagedTensorContext {
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLTensor tensor;
};

inline py::capsule CreateDLPackCapsule(void* data, const std::vector<int64_t>& shape,
                                       DLDataType dtype, int device_id = 0) {
  auto* context = new DLManagedTensorContext();
  context->shape = shape;

  // Calculate strides (row-major)
  context->strides.resize(shape.size());
  int64_t stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    context->strides[i] = stride;
    stride *= shape[i];
  }

  context->tensor.data = data;
  context->tensor.device.device_type = kDLROCM;  // HIP/ROCm device
  context->tensor.device.device_id = device_id;
  context->tensor.ndim = shape.size();
  context->tensor.dtype = dtype;
  context->tensor.shape = context->shape.data();
  context->tensor.strides = context->strides.data();
  context->tensor.byte_offset = 0;

  auto* managed_tensor = new DLManagedTensor();
  managed_tensor->dl_tensor = context->tensor;
  managed_tensor->manager_ctx = context;
  managed_tensor->deleter = [](DLManagedTensor* self) {
    auto* ctx = static_cast<DLManagedTensorContext*>(self->manager_ctx);
    delete ctx;
    delete self;
  };

  // Create a PyCapsule with the DLManagedTensor
  return py::capsule(managed_tensor, "dltensor", [](PyObject* obj) {
    DLManagedTensor* managed = nullptr;
    if (PyCapsule_IsValid(obj, "dltensor")) {
      managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(obj, "dltensor"));
    } else if (PyCapsule_IsValid(obj, "used_dltensor")) {
      managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(obj, "used_dltensor"));
    }

    if (managed && managed->deleter) {
      managed->deleter(managed);
    }
  });
}

// Helper functions to work with DLTensor
inline DLTensor* GetDLTensor(py::capsule capsule) {
  DLManagedTensor* managed_tensor = FromPyCapsule(capsule);
  return &managed_tensor->dl_tensor;
}

inline bool IsDLTensorContiguous(const DLTensor* tensor) {
  // DLPack v1.0+: strides can be nullptr to indicate a compact row-major tensor
  // Check if tensor is contiguous (C-order/row-major)
  if (tensor->strides == nullptr) {
    // nullptr strides means compact row-major (contiguous)
    return true;
  }

  // If strides is not null, verify it represents a contiguous layout
  int64_t expected_stride = 1;
  for (int i = tensor->ndim - 1; i >= 0; --i) {
    if (tensor->strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= tensor->shape[i];
  }
  return true;
}

inline size_t GetDLTensorElementSize(const DLTensor* tensor) {
  return (tensor->dtype.bits + 7) / 8;
}

}  // namespace mori

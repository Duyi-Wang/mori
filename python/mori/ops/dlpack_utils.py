# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""DLPack utilities for interfacing with PyTorch tensors."""

from typing import Any

import torch
from mori import cpp as mori_cpp


def torch_to_dlpack(tensor: torch.Tensor) -> Any:
    """Convert a PyTorch tensor to DLPack capsule.

    Args:
        tensor: A PyTorch tensor (or any tensor implementing __dlpack__ protocol)

    Returns:
        A DLPack capsule that can be passed to C++ code
    """
    if hasattr(tensor, "__dlpack__"):
        return tensor.__dlpack__()
    else:
        raise TypeError(
            f"Object of type {type(tensor)} does not support DLPack protocol"
        )


def dlpack_to_torch(capsule: Any) -> torch.Tensor:
    """Convert a DLPack capsule to PyTorch tensor.

    Args:
        capsule: A DLPack capsule from C++ code

    Returns:
        A PyTorch tensor
    """
    return torch.utils.dlpack.from_dlpack(capsule)


def get_dlpack_dtype(torch_dtype: torch.dtype) -> mori_cpp.DLPackDtype:
    """Convert PyTorch dtype to mori DLPackDtype enum.

    Args:
        torch_dtype: A torch.dtype

    Returns:
        A mori.cpp.DLPackDtype enum value
    """
    dtype_map = {
        torch.float32: mori_cpp.DLPackDtype.Float32,
        torch.float16: mori_cpp.DLPackDtype.Float16,
        torch.bfloat16: mori_cpp.DLPackDtype.BFloat16,
        torch.int32: mori_cpp.DLPackDtype.Int32,
        torch.int64: mori_cpp.DLPackDtype.Int64,
        torch.uint32: mori_cpp.DLPackDtype.UInt32,
        torch.float8_e4m3fnuz: mori_cpp.DLPackDtype.Float8_e4m3fnuz,
    }

    if torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {torch_dtype}")

    return dtype_map[torch_dtype]

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
from __future__ import annotations

from mori import cpp as mori_cpp
from mori.ops.dlpack_utils import torch_to_dlpack, dlpack_to_torch, get_dlpack_dtype

from dataclasses import dataclass
import torch
import torch.distributed as dist


class EpDispatchCombineKernelType(mori_cpp.EpDispatchCombineKernelType):
    def __str__(self):
        return self.name


@dataclass
class EpDispatchCombineConfig:
    data_type: torch.dtype
    rank: int
    world_size: int
    hidden_dim: int
    scale_dim: int
    scale_type_size: int
    max_token_type_size: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    warp_num_per_block: int = 8
    block_num: int = 80
    use_external_inp_buf: bool = True
    kernel_type: EpDispatchCombineKernelType = EpDispatchCombineKernelType.IntraNode
    gpu_per_node: int = 8
    rdma_block_num: int = 0


def _cpp_dispatch_combine_factory(entity_name: str):
    return getattr(mori_cpp, entity_name)


class EpDispatchCombineOp:
    def __init__(self, config: EpDispatchCombineConfig) -> None:
        self.config = config

        handle_class = _cpp_dispatch_combine_factory("EpDispatchCombineHandle")
        self._handle = handle_class(
            mori_cpp.EpDispatchCombineConfig(
                rank=config.rank,
                world_size=config.world_size,
                hidden_dim=config.hidden_dim,
                scale_dim=config.scale_dim,
                scale_type_size=config.scale_type_size,
                max_token_type_size=config.max_token_type_size,
                max_num_inp_token_per_rank=config.max_num_inp_token_per_rank,
                num_experts_per_rank=config.num_experts_per_rank,
                num_experts_per_token=config.num_experts_per_token,
                warp_num_per_block=config.warp_num_per_block,
                block_num=config.block_num,
                use_external_inp_buf=config.use_external_inp_buf,
                gpu_per_node=config.gpu_per_node,
                rdma_block_num=config.rdma_block_num,
            )
        )

        self._dispatch_func = _cpp_dispatch_combine_factory("launch_dispatch")
        self._combine_func = _cpp_dispatch_combine_factory("launch_combine")
        self._reset_func = _cpp_dispatch_combine_factory("launch_reset")
        self._get_dispatch_src_token_pos_func = _cpp_dispatch_combine_factory(
            "get_dispatch_src_token_pos"
        )
        self._get_cur_rank_num_token = _cpp_dispatch_combine_factory(
            "get_cur_rank_num_token"
        )
        self._get_dispatch_sender_token_idx_map_func = _cpp_dispatch_combine_factory(
            "get_dispatch_sender_token_idx_map"
        )
        self._get_dispatch_receiver_token_idx_map_func = _cpp_dispatch_combine_factory(
            "get_dispatch_receiver_token_idx_map"
        )
        self._get_registered_combine_input_buffer = _cpp_dispatch_combine_factory(
            "get_registered_combine_input_buffer"
        )

    def get_registered_combine_input_buffer(self, dtype: torch.dtype) -> torch.Tensor:
        """Get the registered combine input buffer for external use.

        Args:
            dtype: The desired data type for the buffer.

        Returns:
            torch.Tensor: The registered buffer tensor.
        """
        dlpack_dtype = get_dlpack_dtype(dtype)
        capsule = self._get_registered_combine_input_buffer(self._handle, dlpack_dtype)
        return dlpack_to_torch(capsule)

    def dispatch(
        self,
        input: torch.Tensor,
        weights: torch.Tensor | None,
        scales: torch.Tensor | None,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Dispatch tokens to corresponding experts.

        Args:
            input: Input tokens [num_tokens, hidden_dim]
            weights: Routing weights [num_tokens, num_experts_per_token] or None
            scales: Quantization scales [num_tokens, scale_dim] or None
            indices: Expert indices [num_tokens, num_experts_per_token]
            block_num: Number of GPU blocks (optional)
            warp_per_block: Number of warps per block (optional)

        Returns:
            tuple: (out, out_weights, out_scales, out_indices, total_recv_token_num)
                - out: Dispatched tokens [max_tokens_to_recv, hidden_dim]
                - out_weights: Dispatched weights [max_tokens_to_recv, num_experts_per_token] or None
                - out_scales: Dispatched scales [max_tokens_to_recv, scale_dim] or None
                - out_indices: Dispatched indices [max_tokens_to_recv, num_experts_per_token]
                - total_recv_token_num: Actual number of received tokens [1]
        """
        # Convert tensors to DLPack capsules
        input_capsule = torch_to_dlpack(input)
        weights_capsule = torch_to_dlpack(weights) if weights is not None else None
        scales_capsule = torch_to_dlpack(scales) if scales is not None else None
        indices_capsule = torch_to_dlpack(indices)

        # Get current HIP stream from PyTorch
        # Note: PyTorch uses 'cuda_stream' attribute for both CUDA and HIP backends
        current_stream = torch.cuda.current_stream(input.device)
        stream_ptr = current_stream.cuda_stream

        # Call C++ function with DLPack capsules
        result = self._dispatch_func(
            self._handle,
            self.config.kernel_type.value,
            input_capsule,
            weights_capsule,
            scales_capsule,
            indices_capsule,
            block_num,
            warp_per_block,
            stream_ptr,
        )

        # Convert result capsules back to torch tensors
        out = dlpack_to_torch(result[0])
        out_weights = dlpack_to_torch(result[1]) if result[1] is not None else None
        out_scales = dlpack_to_torch(result[2]) if result[2] is not None else None
        out_indices = dlpack_to_torch(result[3])
        total_recv_token_num = dlpack_to_torch(result[4])

        return (out, out_weights, out_scales, out_indices, total_recv_token_num)

    def combine(
        self,
        input: torch.Tensor,
        weights: torch.Tensor | None,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
        call_reset: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Combine expert outputs back to original token order.

        Args:
            input: Expert processed tokens [num_processed_tokens, hidden_dim]
            weights: Routing weights [num_processed_tokens, num_experts_per_token] or None
            indices: Token indices [num_processed_tokens, num_experts_per_token]
            block_num: Number of GPU blocks (optional)
            warp_per_block: Number of warps per block (optional)
            call_reset: Whether to reset state after combine

        Returns:
            tuple: (out, out_weights)
                - out: Combined output tokens [max_num_inp_token_per_rank, hidden_dim]
                - out_weights: Combined weights [max_num_inp_token_per_rank, num_experts_per_token] or None
        """
        # Convert tensors to DLPack capsules
        input_capsule = torch_to_dlpack(input)
        weights_capsule = torch_to_dlpack(weights) if weights is not None else None
        indices_capsule = torch_to_dlpack(indices)

        # Get current HIP stream from PyTorch
        # Note: PyTorch uses 'cuda_stream' attribute for both CUDA and HIP backends
        current_stream = torch.cuda.current_stream(input.device)
        stream_ptr = current_stream.cuda_stream

        # Call C++ function with DLPack capsules
        result = self._combine_func(
            self._handle,
            self.config.kernel_type.value,
            input_capsule,
            weights_capsule,
            indices_capsule,
            block_num,
            warp_per_block,
            stream_ptr,
        )

        if call_reset:
            self._reset_func(self._handle, stream_ptr)

        # Convert result capsules back to torch tensors
        out = dlpack_to_torch(result[0])
        out_weights = dlpack_to_torch(result[1]) if result[1] is not None else None

        return (out, out_weights)

    def reset(self) -> None:
        """Reset internal state for next inference round."""
        self._reset_func(self._handle)

    def _allgather_with_token_num_padding(
        self, input: torch.Tensor, max_token_num: int
    ) -> list[torch.Tensor]:
        shape = list(input.shape)

        pad_shape = shape.copy()
        pad_shape[0] = max_token_num - shape[0]

        target_shape = shape.copy()
        target_shape[0] = max_token_num

        output = [
            torch.zeros(
                target_shape,
                dtype=input.dtype,
                device=input.device,
            )
            for _ in range(self.config.world_size)
        ]
        padded_input = torch.cat(
            [
                input,
                torch.zeros(
                    pad_shape,
                    dtype=input.dtype,
                    device=input.device,
                ),
            ],
            0,
        )
        dist.all_gather(output, padded_input)
        return output

    def get_dispatch_src_token_pos(self) -> torch.Tensor:
        """
        Get the source token positions after dispatch.

        Returns:
            torch.Tensor: Source token positions mapping [num_received_tokens]
        """
        torch.cuda.synchronize()

        if self.config.kernel_type.value in (
            EpDispatchCombineKernelType.IntraNode.value,
            EpDispatchCombineKernelType.InterNodeV1.value,
            EpDispatchCombineKernelType.InterNodeV1LL.value,
        ):
            capsule = self._get_dispatch_src_token_pos_func(self._handle)
            return dlpack_to_torch(capsule)

        dispatch_sender_token_id_map = dlpack_to_torch(
            self._get_dispatch_sender_token_idx_map_func(self._handle)
        )
        dispatch_receiver_token_id_map = dlpack_to_torch(
            self._get_dispatch_receiver_token_idx_map_func(self._handle)
        )

        max_num_token_to_send_per_rank = self.config.max_num_inp_token_per_rank
        all_rank_sender_map = self._allgather_with_token_num_padding(
            dispatch_sender_token_id_map.cpu().to(torch.int64),
            self.config.max_num_inp_token_per_rank * self.config.num_experts_per_token,
        )

        cur_rank_num_token = self._get_cur_rank_num_token(self._handle)
        all_rank_num_token = [torch.empty(1) for i in range(self.config.world_size)]
        dist.all_gather(all_rank_num_token, torch.Tensor([cur_rank_num_token]))

        reverse_sender_token_id_map = {}
        for r in range(self.config.world_size):
            for i, mapped_id in enumerate(
                all_rank_sender_map[r].tolist()[
                    : int(all_rank_num_token[r][0].item())
                    * self.config.num_experts_per_token
                ]
            ):
                dest_pe = mapped_id // max_num_token_to_send_per_rank
                if dest_pe != self.config.rank:
                    continue
                mapped_id = (
                    mapped_id
                    - dest_pe * max_num_token_to_send_per_rank
                    + r * max_num_token_to_send_per_rank
                )
                reverse_sender_token_id_map[mapped_id] = (
                    i // self.config.num_experts_per_token
                )
        src_token_pos = []
        for i, recv_mapped_id in enumerate(dispatch_receiver_token_id_map.tolist()):
            src_pe = recv_mapped_id // max_num_token_to_send_per_rank
            if recv_mapped_id not in reverse_sender_token_id_map:
                print(
                    f"Warning: rank {self.config.rank} src_pe {src_pe} max_num_token_to_send_per_rank {max_num_token_to_send_per_rank} recv_mapped_id {recv_mapped_id} not in reverse_sender_token_id_map"
                )
                raise
            src_tok_id = reverse_sender_token_id_map[recv_mapped_id]
            src_token_pos.append(src_pe * max_num_token_to_send_per_rank + src_tok_id)

        return torch.tensor(src_token_pos, dtype=torch.int)

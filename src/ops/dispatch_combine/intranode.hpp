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

#include "mori/core/core.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void CrossDeviceBarrierIntraNodeKernel(EpDispatchCombineArgs<T> args,
                                                         const uint32_t crossDeviceBarrierFlag) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  if (laneId == 0) atomicAdd(args.combineGridBarrier, 1);

  if (globalThdId < args.config.worldSize) {
    // Set remote flag after all copies are done
    shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, globalWarpNum);
    args.combineGridBarrier[0] = 0;
    core::AtomicStoreRelaxedSystem(
        args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>(globalThdId) + args.config.rank,
        crossDeviceBarrierFlag);
  }

  if (globalThdId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  uint32_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint32_t*>();
  if (thdId < args.config.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchIntraNodeKernel                                   */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpDispatchIntraNodeKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;

  size_t maxNumTokensToSend = config.MaxNumTokensToSend();
  // Sentinel value to mark duplicate tokens (tokens already sent to same PE by earlier expert)
  const index_t DUPLICATE_TOKEN_MARKER = config.worldSize * maxNumTokensToSend;

  // Shared memory to track token counts and starting positions per destination PE
  extern __shared__ char sharedMem[];
  // Size of tokenCountPerPe is: warpNum * npes
  index_t* tokenCountPerPe = reinterpret_cast<index_t*>(sharedMem) + warpId * npes;
  // Start positions in target buffer for each PE
  index_t* startIndexPerPe = reinterpret_cast<index_t*>(sharedMem) + (warpNum + warpId) * npes;

  if (args.tokenIndices && args.inpTokenBuf) {
    // Initialize counters to zero
    for (int pe = laneId; pe < npes; pe += warpSize) {
      tokenCountPerPe[pe] = 0;
      startIndexPerPe[pe] = 0;
    }

    // Step 1: Count tokens going to each destination PE
    for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
         i += globalWarpNum) {
      index_t srcTokId = i / config.numExpertPerToken;
      index_t destExpert = args.tokenIndices[i];
      index_t destPe = destExpert / config.numExpertPerRank;

      // Deduplicate: Skip if this token-expert pair goes to the same PE as an earlier expert
      // for the same token (to avoid sending the same token multiple times to the same PE)
      int condition = 0;
      if (laneId < (i % config.numExpertPerToken)) {
        condition = destPe == (args.tokenIndices[srcTokId * config.numExpertPerToken + laneId] /
                               config.numExpertPerRank);
      }
      if (__any(condition)) {
        // Mark as duplicate so we skip it in the write phase
        if (laneId == 0) args.dispDestTokIdMap[i] = DUPLICATE_TOKEN_MARKER;
        continue;
      }

      // Increment count for this destination PE (lane 0 does the increment)
      if (laneId == 0) {
        tokenCountPerPe[destPe]++;
      }
    }

    // Step 2: Allocate starting positions for each destination PE using a single atomicAdd per PE
    for (int pe = laneId; pe < npes; pe += warpSize) {
      if (tokenCountPerPe[pe] > 0) {
        startIndexPerPe[pe] =
            atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(pe), tokenCountPerPe[pe]);
        atomicAdd(args.destPeTokenCounter + pe, tokenCountPerPe[pe]);
      }
    }

    // Reset counters to track offset within allocated range
    for (int pe = laneId; pe < npes; pe += warpSize) {
      tokenCountPerPe[pe] = 0;
    }

    // Step 3: Write tokens to their allocated positions
    for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
         i += globalWarpNum) {
      index_t srcTokId = i / config.numExpertPerToken;
      index_t destExpert = args.tokenIndices[i];
      index_t destPe = destExpert / config.numExpertPerRank;

      // Skip duplicates (already marked in step 1)
      if (args.dispDestTokIdMap[i] == DUPLICATE_TOKEN_MARKER) {
        continue;
      }

      // Get the destination token ID from start index + current offset
      index_t destTokId = 0;
      if (laneId == 0) {
        destTokId = startIndexPerPe[destPe] + tokenCountPerPe[destPe];
        tokenCountPerPe[destPe]++;
        args.dispDestTokIdMap[i] = destPe * maxNumTokensToSend + destTokId;

        // TODO: use a switch to control the writing of this buffer, should only turn on for testing
        args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
            myPe * config.maxNumInpTokenPerRank + srcTokId;
      }
      destTokId = __shfl(destTokId, 0);

      // Write weights and indices
      if (laneId < config.numExpertPerToken) {
        if (args.weightsBuf) {
          args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
              destPe)[destTokId * config.numExpertPerToken + laneId] =
              args.weightsBuf[srcTokId * config.numExpertPerToken + laneId];
        }
        args.shmemOutIndicesMemObj->template GetAs<index_t*>(
            destPe)[destTokId * config.numExpertPerToken + laneId] =
            args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
      }

      // Write scales
      if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
        index_t destScaleOffset = destTokId * config.scaleDim * config.scaleTypeSize;
        index_t srcScaleOffset = srcTokId * config.scaleDim * config.scaleTypeSize;
        core::WarpCopy(
            args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destScaleOffset,
            args.scalesBuf + srcScaleOffset, config.scaleDim * config.scaleTypeSize);
      }

      index_t srcTokOffset = srcTokId * config.hiddenDim;
      index_t destTokOffset = destTokId * config.hiddenDim;
      core::WarpCopy(args.shmemDispatchOutTokMemObj->template GetAs<T*>(destPe) + destTokOffset,
                     args.inpTokenBuf + srcTokOffset, config.hiddenDim);
    }
  }
  if (laneId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Send token num & token to expert mapping to other ranks
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, globalWarpNum);
      args.dispatchGridBarrier[0] = 0;

      // Send signal to each destPe about how many tokens have been sent to it
      // +1 to distinguish from uninitialized 0
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
  }

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);

      // reset local counter
      args.destPeTokenCounter[destPe] = 0;
      // args.dispatchGridBarrier[destPe] = 0;
    }

    // reset counter
    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineIntraNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__global__ void EpCombineIntraNodeKernel(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;
  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  int globalThdNum = gridDim.x * warpNum * warpSize;

  int myPe = config.rank;
  int npes = config.worldSize;

  const uint32_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];
  size_t maxNumTokensToSend = config.MaxNumTokensToSend();
  // Copy input to shmem registered buffer so that other GPUs can access directly
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  if (args.config.useExternalInpBuffer) {
    for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
      core::WarpCopy(args.shmemCombineInpTokMemObj->template GetAs<T*>() + i * config.hiddenDim,
                     args.inpTokenBuf + i * config.hiddenDim, config.hiddenDim);
    }
  }

  if (args.weightsBuf) {
    for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
      core::WarpCopy(
          args.shmemInpWeightsMemObj->template GetAs<float*>() + i * config.numExpertPerToken,
          args.weightsBuf + i * config.numExpertPerToken, config.numExpertPerToken);
    }
  }

  // Make sure copy on all GPUs are finished
  CrossDeviceBarrierIntraNodeKernel(args, crossDeviceBarrierFlag);
  *args.totalRecvTokenNum = 0;
  if (args.curRankNumToken == 0) return;

  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = reinterpret_cast<float**>(sharedMem) +
                          warpNum * config.numExpertPerToken + warpId * config.numExpertPerToken;

  index_t warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  index_t hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  assert(config.numExpertPerToken < warpSize);
  for (int i = globalWarpId; i < (args.curRankNumToken * warpsPerToken); i += globalWarpNum) {
    index_t tokenId = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;
    index_t hiddenDimOffset = inTokenPartId * hiddenDimPerWarp;
    index_t hiddenDimSize =
        std::max(0, std::min(config.hiddenDim - hiddenDimOffset, hiddenDimPerWarp));

    // Prepare data pointers on different GPUs
    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
      index_t destPe = destTokId / maxNumTokensToSend;

      if (destPe < config.worldSize) {
        index_t destLocalTokId = destTokId - destPe * maxNumTokensToSend;
        srcPtrs[j] = args.shmemCombineInpTokMemObj->template GetAs<T*>(destPe) +
                     destLocalTokId * config.hiddenDim + hiddenDimOffset;
        srcWeightsPtr[j] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                           destLocalTokId * config.numExpertPerToken;
      } else {
        srcPtrs[j] = nullptr;
        srcWeightsPtr[j] = nullptr;
      }
    }
    core::WarpAccum<T, 4>(args.shmemCombineOutTokMemObj->template GetAs<T*>() +
                              tokenId * config.hiddenDim + hiddenDimOffset,
                          srcPtrs, nullptr, config.numExpertPerToken, hiddenDimSize);

    if (args.weightsBuf && inTokenPartId == warpsPerToken - 1) {
      core::WarpAccum<float, 4>(args.shmemCombineOutWeightsMemObj->template GetAs<float*>() +
                                    tokenId * config.numExpertPerToken,
                                srcWeightsPtr, nullptr, config.numExpertPerToken,
                                config.numExpertPerToken);
    }
  }
}

}  // namespace moe
}  // namespace mori

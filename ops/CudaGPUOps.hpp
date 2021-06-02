//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_CUDA_GPU_OPS_HPP
#define ACROBATIC_CUDA_GPU_OPS_HPP

#ifdef ACRO_HAVE_CUDA

#include "NonContractionOps.hpp"
#include "Tensor.hpp"

namespace acro
{


//Internal CPU operations on tensors that are exposed properly by the kernel executors.
//Use of this class directly is not recommended.
class CudaGPUOps : public NonContractionOps
{
    public:
    void BatchMatrixInverse(Tensor &out, Tensor &in);
    void BatchMatrixDet(Tensor &Adet, Tensor &A);
    void BatchMatrixInvDet(Tensor &Ainv, Tensor &Adet, Tensor &A);

    void FlatIndexedScatter(Tensor &Aout, Tensor &Ain, IndexMapping &M);
    void FlatIndexedSumGather(Tensor &Aout, Tensor &Ain, IndexMapping &M);

};


__global__ void CudaInv1x1(float *Ainv, float *A, int N);
__global__ void CudaInv2x2(float *Ainv, float *A, int N);
__global__ void CudaInv3x3(float *Ainv, float *A, int N);
__global__ void CudaDet1x1(float *Adet, float *A, int N);
__global__ void CudaDet2x2(float *Adet, float *A, int N);
__global__ void CudaDet3x3(float *Adet, float *A, int N);
__global__ void CudaInvDet1x1(float *Ainv, float *Adet, float *A, int N);
__global__ void CudaInvDet2x2(float *Ainv, float *Adet, float *A, int N);
__global__ void CudaInvDet3x3(float *Ainv, float *Adet, float *A, int N);
__global__ void CudaScatter(float *Aout, float *Ain, int *M, int *invM, int *invMOff, int N);
__global__ void CudaSumGather(float *Aout, float *Ain, int *M, int *invM, int *invMOff, int N);

}

#endif
#endif //ACROBATIC_CUDA_GPU_OPS_HPP
//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_NATIVE_CPU_OPS_HPP
#define ACROBATIC_NATIVE_CPU_OPS_HPP

#include "NonContractionOps.hpp"
#include "Tensor.hpp"

namespace acro
{


//Internal CPU operations on tensors that are exposed properly by the kernel executors.
//Use of this class directly is not recommended.
class NativeCPUOps : public NonContractionOps
{
    public:
    void BatchMatrixInverse(Tensor &Ainv, Tensor &A);
    void BatchMatrixDet(Tensor &Adet, Tensor &A);
    void BatchMatrixInvDet(Tensor &Ainv, Tensor &Adet, Tensor &A);

    void FlatIndexedScatter(Tensor &Aout, Tensor &Ain, IndexMapping &M);
    void FlatIndexedSumGather(Tensor &Aout, Tensor &Ain, IndexMapping &M);

    private:
    inline void Inv1x1(float *Ainv, float *A, float det);
    inline void Inv2x2(float *Ainv, float *A, float det);
    inline void Inv3x3(float *Ainv, float *A, float det);
    inline float Det1x1(float *A);
    inline float Det2x2(float *A);
    inline float Det3x3(float *A);
};


inline void NativeCPUOps::Inv1x1(float *Ainv, float *A, float det)
{
    Ainv[0] = 1.0 / det;
}


inline void NativeCPUOps::Inv2x2(float *Ainv, float *A, float det)
{
    float invdet = 1.0 / det;
    Ainv[0] = invdet*A[3];
    Ainv[1] = -invdet*A[1];
    Ainv[2] = -invdet*A[2];
    Ainv[3] = invdet*A[0];

}


inline void NativeCPUOps::Inv3x3(float *Ainv, float *A, float det)
{
    float invdet = 1.0 / det;
    Ainv[0] = invdet*(A[4]*A[8] - A[5]*A[7]);
    Ainv[1] = invdet*(A[5]*A[6] - A[3]*A[8]);
    Ainv[2] = invdet*(A[3]*A[7] - A[4]*A[6]);
    Ainv[3] = invdet*(A[2]*A[7] - A[1]*A[8]);
    Ainv[4] = invdet*(A[0]*A[8] - A[2]*A[6]);
    Ainv[5] = invdet*(A[1]*A[6] - A[0]*A[7]);
    Ainv[6] = invdet*(A[1]*A[5] - A[2]*A[4]);
    Ainv[7] = invdet*(A[2]*A[3] - A[0]*A[5]);
    Ainv[8] = invdet*(A[0]*A[4] - A[1]*A[3]);
}


inline float NativeCPUOps::Det1x1(float *A)
{
    return A[0];
}


inline float NativeCPUOps::Det2x2(float *A)
{
    return (A[0]*A[3] - A[1]*A[2]);
}


inline float NativeCPUOps::Det3x3(float *A)
{
    return (A[0]*A[4]*A[8] + A[1]*A[5]*A[6] + A[2]*A[3]*A[7] 
          - A[6]*A[4]*A[2] - A[7]*A[5]*A[0] - A[8]*A[3]*A[1]);
}



}


#endif //ACROBATIC_NATIVE_CPU_OPS_HPP
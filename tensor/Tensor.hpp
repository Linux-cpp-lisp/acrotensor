//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_TENSOR_HPP
#define ACROBATIC_TENSOR_HPP

#include <vector>
#include "Util.hpp"

namespace acro
{

class Tensor
{
    public:
    //Construct and empty tensor to be initilized later
    Tensor();

    //Construct a tensor with the proper dimensions 
    Tensor(std::vector<int> &dims, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    Tensor(int d0, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    Tensor(int d0, int d1, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    Tensor(int d0, int d1, int d2, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    Tensor(int d0, int d1, int d2, int d3, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    Tensor(int d0, int d1, int d2, int d3, int d4, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    Tensor(int d0, int d1, int d2, int d3, int d4, int d5, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    Tensor(int d0, int d1, int d2, int d3, int d4, int d5, int d6, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    Tensor(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    Tensor(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);

    void Init(std::vector<int> &dims, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    void Init(int d0, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    void Init(int d0, int d1, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    void Init(int d0, int d1, int d2, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    void Init(int d0, int d1, int d2, int d3, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    void Init(int d0, int d1, int d2, int d3, int d4, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    void Init(int d0, int d1, int d2, int d3, int d4, int d5, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    void Init(int d0, int d1, int d2, int d3, int d4, int d5, int d6, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    void Init(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);
    void Init(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8, float *hdata=nullptr, float *ddata=nullptr, bool ongpu=false);

    ~Tensor();

    //Simple index into data
    inline float &operator[](int raw_index);

    //Get the simple raw linear index from the tensor indices
    inline int GetRawIndex(const std::vector<int> &indices);
    inline int GetRawIndex(int i0);
    inline int GetRawIndex(int i0, int i1);
    inline int GetRawIndex(int i0, int i1, int i2);
    inline int GetRawIndex(int i0, int i1, int i2, int i3);
    inline int GetRawIndex(int i0, int i1, int i2, int i3, int i4);
    inline int GetRawIndex(int i0, int i1, int i2, int i3, int i4, int i5);
    inline int GetRawIndex(int i0, int i1, int i2, int i3, int i4, int i5, int i6);
    inline int GetRawIndex(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7);
    inline int GetRawIndex(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8);

    //Tensor index into the data
    inline float &operator()(std::vector<int> &indices);
    inline float &operator()(int i0);
    inline float &operator()(int i0, int i1);
    inline float &operator()(int i0, int i1, int i2);
    inline float &operator()(int i0, int i1, int i2, int i3);
    inline float &operator()(int i0, int i1, int i2, int i3, int i4);
    inline float &operator()(int i0, int i1, int i2, int i3, int i4, int i5);
    inline float &operator()(int i0, int i1, int i2, int i3, int i4, int i5, int i6);
    inline float &operator()(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7);
    inline float &operator()(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8);

    //Change the dimensions of the tensor without reorganizing the data representation
    void Reshape(std::vector<int> &dims);
    void Reshape(int d0);
    void Reshape(int d0, int d1);
    void Reshape(int d0, int d1, int d2);
    void Reshape(int d0, int d1, int d2, int d3);
    void Reshape(int d0, int d1, int d2, int d3, int d4);
    void Reshape(int d0, int d1, int d2, int d3, int d4, int d5);
    void Reshape(int d0, int d1, int d2, int d3, int d4, int d5, int d6);
    void Reshape(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7);
    void Reshape(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int i8);
    
    inline int GetRank() const;
    inline int GetSize() const ;
    inline int GetDim(int d) const ;
    inline int GetStride(int d) const;
    virtual float *GetData() const;
    virtual float *GetDeviceData() const;
    inline float *GetCurrentData() const;

    //Change where externally owned data is pointing
    virtual void Retarget(float *hdata, float*ddata=nullptr);

    //Routines for Data on the GPU
    virtual void MapToGPU();            //Allocate memory for the data on the GPU
    virtual void MoveToGPU();           //Copy the data to the GPU and flag the data as currently on the GPU
    virtual void SwitchToGPU();         //Flag the data as currently onGPU
    virtual void UnmapFromGPU();        //Deallocate memory on the GPU
    virtual void MoveFromGPU();         //Copy the data back from the GPU and flag the data as currently on the CPU
    virtual void SwitchFromGPU();       //Flag the data as currently on the CPU
    virtual bool IsMappedToGPU() const {return MappedToGPU;}
    virtual bool IsOnGPU() const {return OnGPU;}
    virtual bool IsInitialized() const {return Initialized;}

    void Set(float val);       //Sets all values in the tensor to a constant
    void Mult(float c);        //Multiply all values by a constant

    void Print();

    protected:
    void UpdateStrides();
    void ComputeSize();

    std::vector<int> Dims;
    std::vector<int> Strides;
    int Size;
    int ByteSize;

    bool Initialized;
    bool OwnsData;
    bool MappedToGPU;
    bool OnGPU;
    float *Data;
    float *DeviceData;
};


inline int Tensor::GetRank() const
{
    return Dims.size();
}


inline int Tensor::GetSize() const
{
    return Size;
}


inline int Tensor::GetDim(int d) const
{
    return Dims[d];
}


inline int Tensor::GetStride(int d) const
{
    return Strides[d];
}


inline float *Tensor::GetData() const
{
    return Data;
}


inline float *Tensor::GetDeviceData() const
{
    return DeviceData;
}


inline float *Tensor::GetCurrentData() const
{
    return (IsOnGPU()) ? DeviceData : Data;
}



inline int Tensor::GetRawIndex(const std::vector<int> &indices)
{
    int index = 0;
    for (unsigned int d = 0; d < indices.size(); ++d)
    {
        index += Strides[d] * indices[d];
    }
    return index;
}


inline int Tensor::GetRawIndex(int i0)
{
    return Strides[0]*i0;
}


inline int Tensor::GetRawIndex(int i0, int i1)
{
    return Strides[0]*i0 + Strides[1]*i1;
}


inline int Tensor::GetRawIndex(int i0, int i1, int i2)
{
    return Strides[0]*i0 + Strides[1]*i1 + Strides[2]*i2;
}


inline int Tensor::GetRawIndex(int i0, int i1, int i2, int i3)
{
    return Strides[0]*i0 + Strides[1]*i1 + Strides[2]*i2 + Strides[3]*i3;
}


inline int Tensor::GetRawIndex(int i0, int i1, int i2, int i3, int i4)
{
    return Strides[0]*i0 + Strides[1]*i1 + Strides[2]*i2 + Strides[3]*i3 + 
           Strides[4]*i4;
}


inline int Tensor::GetRawIndex(int i0, int i1, int i2, int i3, int i4, int i5)
{
    return Strides[0]*i0 + Strides[1]*i1 + Strides[2]*i2 + Strides[3]*i3 + 
           Strides[4]*i4 + Strides[5]*i5;
}


inline int Tensor::GetRawIndex(int i0, int i1, int i2, int i3, int i4, int i5, int i6)
{
    return Strides[0]*i0 + Strides[1]*i1 + Strides[2]*i2 + Strides[3]*i3 + 
           Strides[4]*i4 + Strides[5]*i5 + Strides[6]*i6;
}


inline int Tensor::GetRawIndex(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7)
{
    return Strides[0]*i0 + Strides[1]*i1 + Strides[2]*i2 + Strides[3]*i3 + 
           Strides[4]*i4 + Strides[5]*i5 + Strides[6]*i6 + Strides[7]*i7;
}


inline int Tensor::GetRawIndex(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8)
{
    return Strides[0]*i0 + Strides[1]*i1 + Strides[2]*i2 + Strides[3]*i3 + 
           Strides[4]*i4 + Strides[5]*i5 + Strides[6]*i6 + Strides[7]*i7 +
           Strides[8]*i8;
}


inline float &Tensor::operator()(std::vector<int> &indices) 
{
#if DEBUG
    ACROBATIC_ASSERT(OnGPU, "You have accessed the CPU version of the data that is fresh on the GPU.");
#endif
    return Data[GetRawIndex(indices)];
}

inline float &Tensor::operator()(int i0)
{
#if DEBUG
    ACROBATIC_ASSERT(OnGPU, "You have accessed the CPU version of the data that is fresh on the GPU.");
#endif
    return Data[GetRawIndex(i0)];
}


inline float &Tensor::operator()(int i0, int i1)
{
#if DEBUG
    ACROBATIC_ASSERT(OnGPU, "You have accessed the CPU version of the data that is fresh on the GPU.");
#endif
    return Data[GetRawIndex(i0, i1)];
}


inline float &Tensor::operator[](int raw_index) 
{
#if DEBUG
    ACROBATIC_ASSERT(OnGPU, "You have accessed the CPU version of the data that is fresh on the GPU.");
#endif    
    return Data[raw_index];
}

inline float &Tensor::operator()(int i0, int i1, int i2)
{
#if DEBUG
    ACROBATIC_ASSERT(OnGPU, "You have accessed the CPU version of the data that is fresh on the GPU.");
#endif
    return Data[GetRawIndex(i0, i1, i2)];
}


inline float &Tensor::operator()(int i0, int i1, int i2, int i3)
{
#if DEBUG
    ACROBATIC_ASSERT(OnGPU, "You have accessed the CPU version of the data that is fresh on the GPU.");
#endif
    return Data[GetRawIndex(i0, i1, i2, i3)];
}


inline float &Tensor::operator()(int i0, int i1, int i2, int i3, int i4)
{
#if DEBUG
    ACROBATIC_ASSERT(OnGPU, "You have accessed the CPU version of the data that is fresh on the GPU.");
#endif
    return Data[GetRawIndex(i0, i1, i2, i3, i4)];
}


inline float &Tensor::operator()(int i0, int i1, int i2, int i3, int i4, int i5)
{
#if DEBUG
    ACROBATIC_ASSERT(OnGPU, "You have accessed the CPU version of the data that is fresh on the GPU.");
#endif
    return Data[GetRawIndex(i0, i1, i2, i3, i4, i5)];
}


inline float &Tensor::operator()(int i0, int i1, int i2, int i3, int i4, int i5, int i6)
{
#if DEBUG
    ACROBATIC_ASSERT(OnGPU, "You have accessed the CPU version of the data that is fresh on the GPU.");
#endif
    return Data[GetRawIndex(i0, i1, i2, i3, i4, i5, i6)];
}


inline float &Tensor::operator()(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7)
{
#if DEBUG
    ACROBATIC_ASSERT(OnGPU, "You have accessed the CPU version of the data that is fresh on the GPU.");
#endif
    return Data[GetRawIndex(i0, i1, i2, i3, i4, i5, i6, i7)];
}


inline float &Tensor::operator()(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8)
{
#if DEBUG
    ACROBATIC_ASSERT(OnGPU, "You have accessed the CPU version of the data that is fresh on the GPU.");
#endif
    return Data[GetRawIndex(i0, i1, i2, i3, i4, i5, i6, i7, i8)];
}

}

#endif //ACROBATIC_TENSOR_HPP

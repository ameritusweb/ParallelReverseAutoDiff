//------------------------------------------------------------------------------
// <copyright file="CudaNotInitializedException.cs" author="ameritusweb" date="5/8/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Exceptions
{
    using System;

    /// <summary>
    /// Thrown when a CUDA operation is attempted before the CUDA runtime has been initialized.
    /// Call CudaBlas.Instance.Initialize() to initialize the CUDA runtime.
    /// </summary>
    public class CudaNotInitializedException : Exception
    {
    }
}

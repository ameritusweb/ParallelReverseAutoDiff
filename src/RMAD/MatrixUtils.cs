//------------------------------------------------------------------------------
// <copyright file="MatrixUtils.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    public static class MatrixUtils
    {
        public static Matrix[] Reassemble((Matrix?, Matrix?) dOutput)
        {
            return new Matrix[] { dOutput.Item1, dOutput.Item2 };
        }
    }
}

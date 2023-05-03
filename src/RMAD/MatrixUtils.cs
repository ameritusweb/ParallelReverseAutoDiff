//------------------------------------------------------------------------------
// <copyright file="MatrixUtils.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    public class MatrixUtils
    {
        public static double[][][] Reassemble((double[][]?, double[][]?) dOutput)
        {
            return new double[][][] { dOutput.Item1, dOutput.Item2 };
        }
    }
}

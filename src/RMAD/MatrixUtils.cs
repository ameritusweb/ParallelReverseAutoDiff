using System;
using System.Collections.Generic;
using System.Text;

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

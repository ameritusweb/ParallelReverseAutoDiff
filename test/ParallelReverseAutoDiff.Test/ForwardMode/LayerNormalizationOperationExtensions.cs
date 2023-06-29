using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.ForwardMode
{
    public static class LayerNormalizationOperationExtensions
    {
        public static DualNumberMatrix Forward(this LayerNormalizationOperation operation, DualNumberMatrix input, double epsilon)
        {
            var numRows = input.Length;
            var numCols = input[0].Length;

            var mean = new DualNumber[numRows];
            var stdDev = new DualNumber[numRows];

            // Compute the mean and standard deviation for each row
            for (int i = 0; i < numRows; i++)
            {
                // Compute the mean using the average function, and then manually compute the mean for the dual part
                mean[i] = new DualNumber(input[i].Average(x => x.Real), input[i].Average(x => x.Dual));
                // Compute the standard deviation manually using the sqrt function for DualNumber
                stdDev[i] = DualNumber.Sqrt(DualNumber.Sum(input[i].Select(x => DualNumber.Pow(x - mean[i], 2))) / numCols);
            }

            // Normalize the input
            var Output = new DualNumberMatrix(numRows, numCols);

            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    Output[i][j] = (input[i][j] - mean[i]) / (stdDev[i] + new DualNumber(epsilon, 0));
                }
            });

            return Output;
        }
    }
}

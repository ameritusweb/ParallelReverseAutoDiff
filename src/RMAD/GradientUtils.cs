//------------------------------------------------------------------------------
// <copyright file="GradientUtils.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Gradient utilities for reverse mode automatic differentiation.
    /// </summary>
    public static class GradientUtils
    {
        /// <summary>
        /// Accumulates the gradients for multiple matrices.
        /// </summary>
        /// <param name="gradients">The gradients to accumulate.</param>
        /// <returns>A matrix with the accumulated gradients.</returns>
        public static Matrix? AccumulateGradients(List<Matrix> gradients)
        {
            if (gradients == null || gradients.Count == 0)
            {
                return null;
            }

            int numRows = gradients[0].Length;
            int numCols = gradients[0][0].Length;

            Matrix accumulatedGradients = new Matrix(numRows, numCols);
            Parallel.For(0, numRows, (i) =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    accumulatedGradients[i][j] = gradients.Sum(g => g[i][j]);
                }
            });

            return accumulatedGradients;
        }

        /// <summary>
        /// Accumulates the gradients for multiple deep matrices.
        /// </summary>
        /// <param name="gradients">The gradients to accumulate.</param>
        /// <returns>A deep matrix with the accumulated gradients.</returns>
        public static DeepMatrix? AccumulateGradients(List<DeepMatrix> gradients)
        {
            if (gradients == null || gradients.Count == 0)
            {
                return null;
            }

            int depth = gradients[0].Depth;

            DeepMatrix accumulatedGradients = new DeepMatrix(depth, 1, 1);

            Parallel.For(0, depth, d =>
            {
                int numRows = gradients[0][d].Rows;
                int numCols = gradients[0][d].Cols;

                accumulatedGradients[d] = new Matrix(numRows, numCols);

                for (int i = 0; i < numRows; i++)
                {
                    for (int j = 0; j < numCols; j++)
                    {
                        accumulatedGradients[d][i][j] = gradients.Sum(g => g[d][i][j]);
                    }
                }
            });

            return accumulatedGradients;
        }

        /// <summary>
        /// Accumulate the backward gradients.
        /// </summary>
        /// <param name="gradientsList">The list of gradients to accumulate.</param>
        /// <returns>The accumulated gradients.</returns>
        public static object?[] AccumulateBackwardGradients(List<BackwardResult> gradientsList)
        {
            List<object?> accumulatedBackwardGradients = new List<object?>();
            var firstResult = gradientsList[0];
            var size = firstResult.Results.Length;
            for (int i = 0; i < size; ++i)
            {
                if (firstResult.Results[i] is Matrix)
                {
                    Matrix? accumulatedGradients = AccumulateGradients(gradientsList.Where(g => g.Results[i] != null).Select(g => g.Results[i]).OfType<Matrix>().ToList());
                    accumulatedBackwardGradients.Add(accumulatedGradients);
                }
                else if (firstResult.Results[i] is DeepMatrix)
                {
                    DeepMatrix? accumulatedGradients = AccumulateGradients(gradientsList.Where(g => g.Results[i] != null).Select(g => g.Results[i]).OfType<DeepMatrix>().ToList());
                    accumulatedBackwardGradients.Add(accumulatedGradients);
                }
                else
                {
                    accumulatedBackwardGradients.Add(null);
                }
            }

            return accumulatedBackwardGradients.ToArray();
        }
    }
}

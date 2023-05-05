//------------------------------------------------------------------------------
// <copyright file="GradientUtils.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;
    using System.Linq;

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
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    accumulatedGradients[i][j] = gradients.Sum(g => g[i][j]);
                }
            }

            return accumulatedGradients;
        }

        /// <summary>
        /// Accumulate the backward gradients.
        /// </summary>
        /// <param name="gradientsList">The list of gradients to accumulate.</param>
        /// <returns>The accumulated gradients.</returns>
        public static (Matrix?, Matrix?) AccumulateBackwardGradients(List<(Matrix?, Matrix?)> gradientsList)
        {
            if (gradientsList == null || gradientsList.Count == 0)
            {
                return (null, null);
            }

            List<Matrix> firstGradients = gradientsList.Where(g => g.Item1 != null).Select(g => g.Item1).OfType<Matrix>().ToList();
            List<Matrix> secondGradients = gradientsList.Where(g => g.Item2 != null).Select(g => g.Item2).OfType<Matrix>().ToList();

            Matrix? firstAccumulatedGradients = firstGradients.Count > 0 ? AccumulateGradients(firstGradients) : null;
            Matrix? secondAccumulatedGradients = secondGradients.Count > 0 ? AccumulateGradients(secondGradients) : null;

            return (firstAccumulatedGradients, secondAccumulatedGradients);
        }
    }
}

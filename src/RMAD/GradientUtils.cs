//------------------------------------------------------------------------------
// <copyright file="GradientUtils.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;
    using System.Linq;

    public static class GradientUtils
    {
        public static double[][]? AccumulateGradients(List<double[][]> gradients)
        {
            if (gradients == null || gradients.Count == 0)
            {
                return null;
            }

            int numRows = gradients[0].Length;
            int numCols = gradients[0][0].Length;

            double[][] accumulatedGradients = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                accumulatedGradients[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    accumulatedGradients[i][j] = gradients.Sum(g => g[i][j]);
                }
            }

            return accumulatedGradients;
        }

        public static (double[][]?, double[][]?) AccumulateBackwardGradients(List<(double[][]?, double[][]?)> gradientsList)
        {
            if (gradientsList == null || gradientsList.Count == 0)
            {
                return (null, null);
            }

            List<double[][]> firstGradients = gradientsList.Where(g => g.Item1 != null).Select(g => g.Item1).OfType<double[][]>().ToList();
            List<double[][]> secondGradients = gradientsList.Where(g => g.Item2 != null).Select(g => g.Item2).OfType<double[][]>().ToList();

            double[][]? firstAccumulatedGradients = firstGradients.Count > 0 ? AccumulateGradients(firstGradients) : null;
            double[][]? secondAccumulatedGradients = secondGradients.Count > 0 ? AccumulateGradients(secondGradients) : null;

            return (firstAccumulatedGradients, secondAccumulatedGradients);
        }
    }
}

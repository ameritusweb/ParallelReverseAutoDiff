//------------------------------------------------------------------------------
// <copyright file="GradientClearer.cs" author="ameritusweb" date="5/7/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;

    /// <summary>
    /// A gradient clearer.
    /// </summary>
    public class GradientClearer
    {
        /// <summary>
        /// Clear the gradients from the layers.
        /// </summary>
        /// <param name="layers">The layers.</param>
        public void Clear(IModelLayer[] layers)
        {
            Parallel.For(0, layers.Length, i =>
            {
                var layer = layers[i];
                var identifiers = layer.Identifiers;
                for (int j = 0; j < identifiers.Count; ++j)
                {
                    var identifier = identifiers[j];
                    var gradient = layer[identifier, ModelElementType.Gradient];
                    var dimensions = layer.Dimensions(identifier) ?? throw new InvalidOperationException("Dimensions cannot be null.");
                    switch (dimensions.Length)
                    {
                        case 2:
                            {
                                var gradientMatrix = gradient as Matrix ?? throw new InvalidOperationException("Gradient cannot be null.");
                                this.ClearMatrix(gradientMatrix);
                                break;
                            }

                        case 3:
                            {
                                var gradientMatrix = gradient as DeepMatrix ?? throw new InvalidOperationException("Gradient cannot be null.");
                                this.ClearDeepMatrix(gradientMatrix);
                                break;
                            }

                        case 4:
                            {
                                var gradientMatrix = gradient as DeepMatrix[] ?? throw new InvalidOperationException("Gradient cannot be null.");
                                for (int d = 0; d < dimensions[0]; ++d)
                                {
                                    this.ClearDeepMatrix(gradientMatrix[d]);
                                }

                                break;
                            }
                    }
                }
            });
        }

        private void ClearDeepMatrix(DeepMatrix matrices)
        {
            int numMatrices = matrices.Depth;

            // Parallelize the outer loop
            Parallel.For(0, numMatrices, i =>
            {
                int numRows = matrices[i].Length;
                int numCols = matrices[i][0].Length;
                for (int j = 0; j < numRows; ++j)
                {
                    matrices[i][j] = new double[numCols];
                }
            });
        }

        private void ClearMatrix(Matrix matrix)
        {
            // Parallelize the outer loop
            Parallel.For(0, matrix.Rows, i =>
            {
                matrix[i] = new double[matrix.Cols];
            });
        }
    }
}

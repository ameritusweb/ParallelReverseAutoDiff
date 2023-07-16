// ------------------------------------------------------------------------------
// <copyright file="CosineDistanceLossOperation.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    /// <summary>
    /// Cosine distance loss operation.
    /// </summary>
    public class CosineDistanceLossOperation
    {
        private Matrix superpath;
        private Matrix targetPath;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static CosineDistanceLossOperation Instantiate(NeuralNetwork net)
        {
            return new CosineDistanceLossOperation();
        }

        /// <summary>
        /// Performs the forward operation for the cosine distance loss function.
        /// </summary>
        /// <param name="superpath">The superpath matrix.</param>
        /// <param name="targetPath">The target path matrix.</param>
        /// <returns>The scalar loss value.</returns>
        public Matrix Forward(Matrix superpath, Matrix targetPath)
        {
            this.superpath = superpath;
            this.targetPath = targetPath;

            // Calculate cosine similarity between superpath and target path
            double cosineSimilarity = superpath.CosineSimilarity(targetPath);

            // Calculate loss as 1 - cosine similarity (i.e., cosine distance)
            double loss = 1 - cosineSimilarity;

            var output = new Matrix(1, 1);
            output[0][0] = loss;

            return output;
        }

        /// <summary>
        /// Runs the backward operation for the cosine distance loss function.
        /// </summary>
        /// <param name="dOutput">The gradient of the output.</param>
        /// <param name="targetMatrix">The target matrix.</param>
        /// <returns>The backward result.</returns>
        public BackwardResult Backward(Matrix dOutput, Matrix? targetMatrix)
        {
            double dLoss = targetMatrix == null ? 1d : -1d;
            Matrix dSuperpath = this.superpath.GradientWRTCosineSimilarity(targetMatrix == null ? this.targetPath : targetMatrix, dLoss);

            return new BackwardResultBuilder()
                .AddInputGradient(dSuperpath)
                .Build();
        }
    }
}
//------------------------------------------------------------------------------
// <copyright file="DeepScaleAndShiftOperation.cs" author="ameritusweb" date="5/14/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Threading.Tasks;

    /// <summary>
    /// Deep scale and shift operation.
    /// </summary>
    public class DeepScaleAndShiftOperation : DeepOperation
    {
        private DeepMatrix input;
        private Matrix beta;

        /// <summary>
        /// Gets the gradient of beta.
        /// </summary>
        public Matrix GradientBeta { get; private set; }

        /// <summary>
        /// Gets the gradient of gamma.
        /// </summary>
        public Matrix GradientGamma { get; private set; }

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IDeepOperation Instantiate(NeuralNetwork net)
        {
            return new DeepScaleAndShiftOperation();
        }

        /// <summary>
        /// The forward pass of the scale and shift operation.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <param name="beta">The beta matrix.</param>
        /// <param name="gamma">The gamma matrix.</param>
        /// <returns>The output matrix.</returns>
        public DeepMatrix Forward(DeepMatrix input, Matrix beta, Matrix gamma)
        {
            this.input = input;
            this.beta = beta;

            int depth = input.Depth;
            int numRows = input.Rows;
            int numCols = input.Cols;

            this.DeepOutput = new DeepMatrix(depth, numRows, numCols);

            Parallel.For(0, depth, d =>
            {
                for (int i = 0; i < numRows; i++)
                {
                    for (int j = 0; j < numCols; j++)
                    {
                        this.DeepOutput[d, i, j] = (input[d, i, j] * beta[i, d]) + gamma[i, d];
                    }
                }
            });

            return this.DeepOutput;
        }

        /// <summary>
        /// Calculates the gradient of the scale and shift operation with respect to the input, beta, and gamma matrices.
        /// </summary>
        /// <param name="gradOutput">The gradient of the output matrix.</param>
        /// <returns>A tuple containing the gradients for the input, beta, and gamma matrices.</returns>
        public override BackwardResult Backward(DeepMatrix gradOutput)
        {
            int depth = this.input.Depth;
            int numRows = this.input.Rows;
            int numCols = this.input.Cols;

            DeepMatrix dInput = new DeepMatrix(depth, numRows, numCols);
            Matrix dBeta = new Matrix(numRows, depth);
            Matrix dGamma = new Matrix(numRows, depth);

            Parallel.For(0, depth, d =>
            {
                for (int i = 0; i < numRows; i++)
                {
                    dBeta[i, d] = 0;
                    dGamma[i, d] = 0;

                    for (int j = 0; j < numCols; j++)
                    {
                        dInput[d, i, j] = gradOutput[d, i, j] * this.beta[i, d];
                        dBeta[i, d] += gradOutput[d, i, j] * this.input[d, i, j];
                        dGamma[i, d] += gradOutput[d, i, j];
                    }
                }
            });

            this.GradientBeta = dBeta;
            this.GradientGamma = dGamma;

            return new BackwardResult { DeepInputGradient = dInput, BetaGradient = dBeta, GammaGradient = dGamma };
        }
    }
}

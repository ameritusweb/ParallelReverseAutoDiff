//------------------------------------------------------------------------------
// <copyright file="ScaleAndShiftOperation.cs" author="ameritusweb" date="5/8/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Threading.Tasks;

    /// <summary>
    /// Scale and shift operation.
    /// </summary>
    public class ScaleAndShiftOperation : Operation
    {
        private Matrix input;
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
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ScaleAndShiftOperation();
        }

        /// <summary>
        /// The forward pass of the scale and shift operation.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <param name="beta">The beta matrix.</param>
        /// <param name="gamma">The gamma matrix.</param>
        /// <returns>The output matrix.</returns>
        public Matrix Forward(Matrix input, Matrix beta, Matrix gamma)
        {
            this.input = input;
            this.beta = beta;

            int numRows = input.Length;
            int numCols = input[0].Length;

            this.Output = new Matrix(numRows, numCols);

            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    this.Output[i][j] = (input[i][j] * beta[i][0]) + gamma[i][0];
                }
            });

            return this.Output;
        }

        /// <summary>
        /// Calculates the gradient of the scale and shift operation with respect to the input, beta, and gamma matrices.
        /// </summary>
        /// <param name="gradOutput">The gradient of the output matrix.</param>
        /// <returns>A tuple containing the gradients for the input, beta, and gamma matrices.</returns>
        public override BackwardResult Backward(Matrix gradOutput)
        {
            int numRows = this.input.Length;
            int numCols = this.input[0].Length;

            Matrix gradientInput = new Matrix(numRows, numCols);
            Matrix gradientBeta = new Matrix(numRows, 1);
            Matrix gradientGamma = new Matrix(numRows, 1);

            Parallel.For(0, numRows, i =>
            {
                gradientBeta[i][0] = 0;
                gradientGamma[i][0] = 0;

                for (int j = 0; j < numCols; j++)
                {
                    gradientInput[i][j] = gradOutput[i][j] * this.beta[i][0];
                    gradientBeta[i][0] += gradOutput[i][j] * this.input[i][j];
                    gradientGamma[i][0] += gradOutput[i][j];
                }
            });

            this.GradientBeta = gradientBeta;
            this.GradientGamma = gradientGamma;

            return new BackwardResult() { InputGradient = gradientInput, BetaGradient = gradientBeta, GammaGradient = gradientGamma };
        }
    }
}

//------------------------------------------------------------------------------
// <copyright file="LeakyReLUOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// A leaky ReLU operation.
    /// </summary>
    public class LeakyReLUOperation : Operation
    {
        private readonly double alpha;
        private Matrix input;

        /// <summary>
        /// Initializes a new instance of the <see cref="LeakyReLUOperation"/> class.
        /// </summary>
        /// <param name="alpha">The alpha.</param>
        public LeakyReLUOperation(double alpha)
        {
            this.alpha = alpha;
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new LeakyReLUOperation(net.Parameters.LeakyReLUAlpha);
        }

        /// <summary>
        /// The forward pass of the leaky ReLU operation.
        /// </summary>
        /// <param name="input">The input for the leaky ReLU operation.</param>
        /// <returns>The output for the leaky ReLU operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int rows = input.Length;
            int cols = input[0].Length;
            this.Output = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = input[i][j];
                    this.Output[i][j] = x > 0 ? x : this.alpha * x;
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int rows = dLdOutput.Length;
            int cols = dLdOutput[0].Length;
            Matrix dLdInput = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = this.input[i][j];
                    double gradient = x > 0 ? 1.0 : this.alpha;
                    dLdInput[i][j] = dLdOutput[i][j] * gradient;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }
    }
}

//------------------------------------------------------------------------------
// <copyright file="ReLUOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// Performs the forward and backward operations for the ReLU activation function.
    /// </summary>
    public class ReLUOperation : Operation
    {
        private double[][] input;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReLUOperation"/> class.
        /// </summary>
        public ReLUOperation()
            : base()
        {
        }

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ReLUOperation();
        }

        /// <summary>
        /// Performs the forward operation for the ReLU activation function.
        /// </summary>
        /// <param name="input">The input to the ReLU operation.</param>
        /// <returns>The output of the ReLU operation.</returns>
        public double[][] Forward(double[][] input)
        {
            this.input = input;
            int rows = input.Length;
            int cols = input[0].Length;
            this.output = new double[rows][];

            for (int i = 0; i < rows; i++)
            {
                this.output[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    double x = input[i][j];
                    this.output[i][j] = x > 0 ? x : 0;
                }
            }

            return this.output;
        }

        /// <inheritdoc />
        public override (double[][]?, double[][]?) Backward(double[][] dLdOutput)
        {
            int rows = dLdOutput.Length;
            int cols = dLdOutput[0].Length;
            double[][] dLdInput = new double[rows][];

            for (int i = 0; i < rows; i++)
            {
                dLdInput[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    double x = this.input[i][j];
                    double gradient = x > 0 ? 1.0 : 0.0;
                    dLdInput[i][j] = dLdOutput[i][j] * gradient;
                }
            }

            return (dLdInput, dLdInput);
        }
    }
}

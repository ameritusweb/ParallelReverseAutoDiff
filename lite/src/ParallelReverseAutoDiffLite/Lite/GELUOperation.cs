//------------------------------------------------------------------------------
// <copyright file="GELUOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Performs the forward and backward operations for the GELU activation function.
    /// </summary>
    public class GELUOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new GELUOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrices.AddOrUpdate(id, this.input, (x, y) => this.input);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.input = this.IntermediateMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation for the GELU activation function.
        /// </summary>
        /// <param name="input">The input to the GELU operation.</param>
        /// <returns>The output of the GELU operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int rows = input.Rows;
            int cols = input.Cols;

            this.Output = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float x = input[i, j];
                    float gelu = this.GELU(x);
                    this.Output[i, j] = gelu;
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int rows = dLdOutput.Rows;
            int cols = dLdOutput.Cols;
            Matrix dLdInput = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float x = this.input[i, j];
                    dLdInput[i, j] = dLdOutput[i, j] * this.DerivativeGELU(x);
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }

        private float GELU(float x)
        {
            float a = 0.5f * (1.0f + PradMath.Tanh(
                PradMath.Sqrt(2 / PradMath.PI) * (x + (0.044715f * PradMath.Pow(x, 3)))));

            return a * x;
        }

        private float DerivativeGELU(float x)
        {
            float b = PradMath.Tanh(PradMath.Sqrt(2 / PradMath.PI) * (x + (0.044715f * PradMath.Pow(x, 3))));
            float c = PradMath.Pow(1 - PradMath.Pow(b, 2), 2);
            float d = PradMath.Sqrt(2f / PradMath.PI) * ((0.044715f * 3 * PradMath.Pow(x, 2)) + 1);

            return (0.5f * x * ((c * d) + b)) + (0.5f * (1 + b));
        }
    }
}

//------------------------------------------------------------------------------
// <copyright file="PiecewiseActivationOperation.cs" author="ameritusweb" date="12/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// A piece-wise activation operation.
    /// </summary>
    public class PiecewiseActivationOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static PiecewiseActivationOperation Instantiate(NeuralNetwork net)
        {
            return new PiecewiseActivationOperation();
        }

        /// <summary>
        /// Applies a custom element-wise transformation to the input matrix.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <returns>The transformed matrix.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int numRows = input.Rows;
            int numCols = input.Cols;

            this.Output = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    double x = input[i, j];
                    double transformedValue;

                    if (x < 1.63)
                    {
                        transformedValue = (3 / (1 + Math.Exp(-x))) - 1.5;
                    }
                    else if (x < 18.7)
                    {
                        transformedValue = (0.4119 * x) + 0.3372;
                    }
                    else
                    {
                        transformedValue = (7 / (1 + Math.Exp(16 - x))) + 1.48;
                    }

                    this.Output[i, j] = transformedValue;
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int numRows = this.input.Rows;
            int numCols = this.input.Cols;
            Matrix dLdInput = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    double x = this.input[i, j];
                    double gradient;

                    if (x < 1.63)
                    {
                        double expNegX = Math.Exp(-x);
                        gradient = 3 * expNegX / Math.Pow(1 + expNegX, 2);

                        if (double.IsNaN(gradient) || double.IsInfinity(gradient))
                        {
                            throw new InvalidOperationException($"NaN or Infinity encountered in gradient: {x} {expNegX} {i} {j}");
                        }
                    }
                    else if (x < 18.7)
                    {
                        gradient = 0.4119;
                    }
                    else
                    {
                        double exp16MinusX = Math.Exp(16 - x);
                        gradient = -7 * exp16MinusX / Math.Pow(1 + exp16MinusX, 2);

                        if (double.IsNaN(gradient) || double.IsInfinity(gradient))
                        {
                            throw new InvalidOperationException($"NaN or Infinity encountered in gradient: {x} {exp16MinusX} {i} {j}");
                        }
                    }

                    dLdInput[i, j] = dLdOutput[i, j] * gradient;

                    if (double.IsNaN(dLdInput[i, j]) || double.IsInfinity(dLdInput[i, j]))
                    {
                        throw new InvalidOperationException($"NaN or Infinity encountered in dLdInput: {dLdOutput[i, j]} {gradient} {i} {j}");
                    }
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }
    }
}

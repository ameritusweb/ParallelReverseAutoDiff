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
                    float x = input[i, j];
                    float transformedValue;

                    if (x < 1.63)
                    {
                        transformedValue = (3 / (1 + PradMath.Exp(-x))) - 1.5f;
                    }
                    else if (x < 18.7)
                    {
                        transformedValue = (0.4119f * x) + 0.3372f;
                    }
                    else
                    {
                        transformedValue = (7 / (1 + PradMath.Exp(16 - x))) + 1.48f;
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
                    float x = this.input[i, j];
                    float gradient;

                    if (x < -100)
                    {
                        gradient = float.Epsilon;
                    }
                    else if (x < 1.63)
                    {
                        float expNegX = PradMath.Exp(-x);
                        gradient = 3 * expNegX / PradMath.Pow(1 + expNegX, 2);

                        if (float.IsNaN(gradient) || float.IsInfinity(gradient))
                        {
                            throw new InvalidOperationException($"NaN or Infinity encountered in gradient: {x} {expNegX} {i} {j}");
                        }
                    }
                    else if (x < 18.7)
                    {
                        gradient = 0.4119f;
                    }
                    else
                    {
                        float exp16MinusX = PradMath.Exp(16 - x);
                        gradient = -7 * exp16MinusX / PradMath.Pow(1 + exp16MinusX, 2);

                        if (float.IsNaN(gradient) || float.IsInfinity(gradient))
                        {
                            throw new InvalidOperationException($"NaN or Infinity encountered in gradient: {x} {exp16MinusX} {i} {j}");
                        }
                    }

                    dLdInput[i, j] = dLdOutput[i, j] * gradient;

                    if (float.IsNaN(dLdInput[i, j]) || float.IsInfinity(dLdInput[i, j]))
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

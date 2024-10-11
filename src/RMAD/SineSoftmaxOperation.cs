//------------------------------------------------------------------------------
// <copyright file="SineSoftmaxOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Sine Softmax operation.
    /// </summary>
    public class SineSoftmaxOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SineSoftmaxOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.input, this.Output }, (x, y) => new[] { this.input, this.Output });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.input = restored[0];
            this.Output = restored[1];
        }

        /// <summary>
        /// Performs the forward operation for the softmax function.
        /// </summary>
        /// <param name="input">The input to the softmax operation.</param>
        /// <returns>The output of the softmax operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            this.Output = this.SineSoftmax(input);
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int numRows = this.Output.Length;
            int numCols = this.Output[0].Length;

            Matrix dLdInput = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                var mM = PradTools.Zero;
                for (int k = 0; k < numCols; k++)
                {
                    mM += PradMath.Exp(PradMath.Sin(this.input[i, k]));
                }

                for (int j = 0; j < numCols; j++)
                {
                    var dSinSoftmaxj = (mM * PradMath.Exp(PradMath.Sin(this.input[i, j])) * PradMath.Cos(this.input[i, j])) / PradMath.Pow(mM + PradMath.Exp(PradMath.Sin(this.input[i, j])), 2);
                    dLdInput[i][j] = dLdOutput[i][j] * dSinSoftmaxj;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }

        /// <summary>
        /// Computes the gradient of the SineSoftmax output with respect to the input using RMAD.
        /// This is the backward pass only, calculating exp(sin(x)) during the process, and multiplying by the upstream gradient.
        /// </summary>
        /// <param name="input">The input matrix for which the gradients are being computed.</param>
        /// <param name="upstreamGradient">The upstream gradient matrix flowing from the next layer.</param>
        /// <returns>The gradient of the output w.r.t the input.</returns>
        public Matrix SineSoftmaxBackward(Matrix input, Matrix upstreamGradient)
        {
            int numRows = input.Rows;
            int numCols = input.Cols;

            Matrix gradient = new Matrix(numRows, numCols); // The final gradient matrix

            for (int i = 0; i < numRows; i++)
            {
                double mM = 0.0;

                // First, we need to compute the sum over the entire row (denominator) mM
                double[] expSineVals = new double[numCols]; // Cache the exp(sin(x)) values

                for (int j = 0; j < numCols; j++)
                {
                    double sineValue = PradMath.Sin(input[i, j]);      // Calculate sin(x)
                    expSineVals[j] = PradMath.Exp(sineValue);          // Calculate and store exp(sin(x))
                    mM += expSineVals[j];                              // Accumulate mM over the whole row
                }

                // Reverse pass: Calculate gradients w.r.t each input (numerator and denominator)
                for (int j = 0; j < numCols; j++)
                {
                    double expSineVal = expSineVals[j];                // Reuse the cached exp(sin(x))
                    double denominator = mM + expSineVal;                           // Denominator is the sum over the whole row

                    // Gradient w.r.t numerator
                    double gradNumerator = 1.0 / denominator;

                    // Gradient w.r.t denominator
                    double gradDenominator = -expSineVal / (denominator * denominator);

                    // Gradient of exp(sin(x)) is cos(x) * exp(sin(x))
                    double gradExpSin = PradMath.Cos(input[i, j]) * expSineVal;

                    // Combine with upstream gradient
                    double upstreamGrad = upstreamGradient[i, j];      // Upstream gradient for this element

                    double passN = upstreamGrad * gradNumerator;

                    double passD = upstreamGrad * gradDenominator;

                    double accum1 = passN * gradExpSin;

                    double accum2 = passD * gradExpSin;

                    double accum3 = passD * gradExpSin;

                    // 1. Accumulation of gradient w.r.t the numerator
                    gradient[i, j] = PradTools.Cast(upstreamGrad * gradNumerator * gradExpSin);

                    // 2. Accumulation of gradient w.r.t the summation term of the denominator
                    gradient[i, j] += PradTools.Cast(upstreamGrad * gradDenominator * gradExpSin);

                    // 3. Accumulation of gradient w.r.t the current element's contribution to the denominator
                    gradient[i, j] += PradTools.Cast(upstreamGrad * gradDenominator * gradExpSin);
                }
            }

            return gradient;
        }

        private Matrix SineSoftmax(Matrix input)
        {
            int numRows = input.Rows;
            int numCols = input.Cols;

            Matrix output = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                var mM = PradTools.Zero;

                for (int j = 0; j < numCols; j++)
                {
                    mM += PradMath.Exp(PradMath.Sin(input[i, j]));
                }

                for (int j = 0; j < numCols; j++)
                {
                    var numerator = PradMath.Exp(PradMath.Sin(input[i, j]));
                    output[i, j] = numerator / (mM + numerator);
                }
            }

            return output;
        }
    }
}

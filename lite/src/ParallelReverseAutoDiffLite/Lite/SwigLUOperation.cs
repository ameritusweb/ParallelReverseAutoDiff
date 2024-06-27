//------------------------------------------------------------------------------
// <copyright file="SwigLUOperation.cs" author="ameritusweb" date="5/16/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// SwigLU operation for a Matrix.
    /// </summary>
    public class SwigLUOperation : Operation, IOperation
    {
        private readonly float beta;
        private Matrix input;
        private Matrix w;
        private Matrix v;
        private Matrix b;
        private Matrix c;

        /// <summary>
        /// Initializes a new instance of the <see cref="SwigLUOperation"/> class.
        /// </summary>
        /// <param name="beta">beta.</param>
        public SwigLUOperation(float beta)
        {
            this.beta = beta;
        }

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SwigLUOperation(net.Parameters.SwigLUBeta);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.input, this.w, this.v, this.b, this.c }, (x, y) => new[] { this.input, this.w, this.v, this.b, this.c });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.input = restored[0];
            this.w = restored[1];
            this.v = restored[2];
            this.b = restored[3];
            this.c = restored[4];
        }

        /// <summary>
        /// The forward pass of the SwigLU operation.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <param name="w">Weight w.</param>
        /// <param name="v">Weight v.</param>
        /// <param name="b">Bias b.</param>
        /// <param name="c">Bias c.</param>
        /// <returns>The output matrix.</returns>
        public Matrix Forward(Matrix input, Matrix w, Matrix v, Matrix b, Matrix c)
        {
            this.input = input;
            this.w = w;
            this.v = v;
            this.b = b;
            this.c = c;
            int rows = input.Rows;
            int cols = input.Cols;

            this.Output = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float x = input[i, j];
                    float swish = ((x * this.w[j, 0]) + this.b[j][0]) * (1 / (1 + PradMath.Exp(-this.beta * ((x * this.w[j, 0]) + this.b[j][0]))));
                    this.Output[i, j] = swish * ((x * this.v[j, 0]) + this.c[j][0]);
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int rows = this.input.Rows;
            int cols = this.input.Cols;

            Matrix dInput = new Matrix(rows, cols);
            Matrix dW = new Matrix(cols, 1);
            Matrix dV = new Matrix(cols, 1);
            Matrix db = new Matrix(cols, 1);
            Matrix dc = new Matrix(cols, 1);

            for (int j = 0; j < cols; j++)
            {
                float sumdW = 0;
                float sumdV = 0;
                float sumdb = 0;
                float sumdc = 0;

                for (int i = 0; i < rows; i++)
                {
                    float x = this.input[i, j];
                    float f = (x * this.w[j, 0]) + this.b[j][0];
                    float sigmoid_f = 1 / (1 + PradMath.Exp(-this.beta * f));
                    float swish = f * sigmoid_f;  // Swish activation function defined as f * sigmoid(f)
                    float swishDerivative = swish + (this.beta * sigmoid_f * (1 - swish));

                    float dOutputij = dOutput[i, j];

                    dInput[i, j] = dOutputij * (this.v[j, 0] * swishDerivative);

                    sumdW += dOutputij * ((x * swishDerivative) * ((x * this.v[j, 0]) + this.c[j, 0]));

                    sumdV += dOutputij * (swish * x);

                    sumdb += dOutputij * (swishDerivative * ((x * this.v[j, 0]) + this.c[j, 0]));

                    sumdc += dOutputij * swish;
                }

                dW[j, 0] = sumdW;
                dV[j, 0] = sumdV;
                db[j, 0] = sumdb;
                dc[j, 0] = sumdc;
            }

            BackwardResult result = new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .AddWeightGradient(dW)
                .AddWeightGradient(dV)
                .AddBiasGradient(db)
                .AddBiasGradient(dc)
                .Build();

            return result;
        }
    }
}

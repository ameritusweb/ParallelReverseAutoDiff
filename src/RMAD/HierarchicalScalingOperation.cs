//------------------------------------------------------------------------------
// <copyright file="HierarchicalScalingOperation.cs" author="ameritusweb" date="9/24/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Hierarchical scaling operation.
    /// </summary>
    public class HierarchicalScalingOperation : Operation
    {
        private Matrix originalInput;
        private Matrix input;
        private Matrix rowScalars;    // Vector containing row scalars
        private Matrix colScalars;    // Vector containing column scalars
        private Matrix weightMatrix;  // Matrix for full scaling

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new HierarchicalScalingOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.originalInput, this.input, this.rowScalars, this.colScalars, this.weightMatrix }, (_, _) => new[] { this.originalInput, this.input, this.rowScalars, this.colScalars, this.weightMatrix });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.originalInput = restored[0];
            this.input = restored[1];
            this.rowScalars = restored[2];
            this.colScalars = restored[3];
            this.weightMatrix = restored[4];
        }

        /// <summary>
        /// Performs the forward operation for the hierarchical scaling function.
        /// </summary>
        /// <param name="input">The input matrix. (MxN).</param>
        /// <param name="rowScalars">The row scalars. (Mx1).</param>
        /// <param name="colScalers">The column scalars. (1xN).</param>
        /// <param name="weightMatrix">The weight matrix. (MxN).</param>
        /// <returns>The resultant matrix. (MxN).</returns>
        public Matrix Forward(Matrix input, Matrix rowScalars, Matrix colScalers, Matrix weightMatrix)
        {
            this.originalInput = (Matrix)input.Clone();
            this.input = input;
            this.rowScalars = rowScalars;
            this.colScalars = colScalers;
            this.weightMatrix = weightMatrix;

            // Step 1: Row-wise scaling
            for (int i = 0; i < input.Rows; i++)
            {
                for (int j = 0; j < input.Cols; j++)
                {
                    input[i][j] *= this.SafeExponential(rowScalars[i][0]);
                }
            }

            // Step 2: Column-wise scaling
            for (int j = 0; j < input.Cols; j++)
            {
                for (int i = 0; i < input.Rows; i++)
                {
                    input[i][j] *= this.SafeExponential(this.colScalars[0][j]);
                }
            }

            // Step 3: Full scaling
            this.Output = input.ElementwiseMultiply(weightMatrix.ExponentialElementwise());

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            // Initialize matrices for gradients
            Matrix dInputColScaled = new Matrix(this.input.Rows, this.input.Cols);
            Matrix dInputRowScaled = new Matrix(this.input.Rows, this.input.Cols);
            Matrix dInput = new Matrix(this.input.Rows, this.input.Cols);

            Matrix dRowScalars = new Matrix(this.input.Rows, 1); // Corrected to column vector
            Matrix dColScalars = new Matrix(1, this.input.Cols);  // Assuming row vector

            // Gradient from WeightMatrix scaling
            for (int i = 0; i < this.input.Rows; i++)
            {
                for (int j = 0; j < this.input.Cols; j++)
                {
                    dInputColScaled[i][j] = dOutput[i][j] * this.SafeExponential(this.weightMatrix[i][j]) * this.input[i][j];
                }
            }

            // Recompute inputRowScaled
            Matrix inputRowScaled = (Matrix)this.originalInput.Clone();
            for (int i = 0; i < this.input.Rows; i++)
            {
                for (int j = 0; j < this.input.Cols; j++)
                {
                    inputRowScaled[i][j] *= this.rowScalars[i][0];
                }
            }

            // Gradient from column scaling
            for (int j = 0; j < this.input.Cols; j++)
            {
                double colGrad = 0;
                for (int i = 0; i < this.input.Rows; i++)
                {
                    var exp = this.SafeExponential(this.colScalars[0][j]);
                    dInputRowScaled[i][j] = dInputColScaled[i][j] * exp;
                    colGrad += dInputColScaled[i][j] * inputRowScaled[i][j] * exp;
                }

                dColScalars[0][j] = colGrad;
            }

            // Gradient from row scaling
            for (int i = 0; i < this.input.Rows; i++)
            {
                double rowGrad = 0;
                for (int j = 0; j < this.input.Cols; j++)
                {
                    var exp = this.SafeExponential(this.rowScalars[i][0]);
                    dInput[i][j] = dInputRowScaled[i][j] * exp;
                    rowGrad += dInputRowScaled[i][j] * this.originalInput[i][j] * exp;
                }

                dRowScalars[i][0] = rowGrad;
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .AddInputGradient(dRowScalars)
                .AddInputGradient(dColScalars)
                .AddInputGradient(dInputColScaled)
                .Build();
        }

        private double SafeExponential(double x)
        {
            if (x > Math.Log(double.MaxValue))
            {
                return double.MaxValue;
            }
            else
            {
                return Math.Exp(x);
            }
        }
    }
}

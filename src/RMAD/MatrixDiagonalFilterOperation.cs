//------------------------------------------------------------------------------
// <copyright file="MatrixDiagonalFilterOperation.cs" author="ameritusweb" date="12/13/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Matrix diagonal filter operation.
    /// </summary>
    public class MatrixDiagonalFilterOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static MatrixDiagonalFilterOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixDiagonalFilterOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.input1, this.input2 }, (x, y) => new[] { this.input1, this.input2 });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.input1 = restored[0];
            this.input2 = restored[1];
        }

        /// <summary>
        /// Performs the forward operation for the matrix diagonal filter function.
        /// </summary>
        /// <param name="input1">The first input to the matrix diagonal filter operation.</param>
        /// <param name="input2">The second input to the matrix diagonal filter operation.</param>
        /// <returns>The output of the matrix diagonal filter operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2)
        {
            this.input1 = input1;
            this.input2 = input2;

            int n = input1.Rows;
            if (input1.Cols != n || input2.Rows != n || input2.Cols != n)
            {
                throw new ArgumentException("Both matrices must be square and of the same size.");
            }

            this.Output = new Matrix(n, n);

            Parallel.For(0, n, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j)
                    {
                        this.Output[i][j] = 0; // Diagonal elements set to zero
                    }
                    else if (i < j)
                    {
                        this.Output[i][j] = input1[i][j]; // Values above the diagonal from input1
                    }
                    else
                    {
                        this.Output[i][j] = input2[i][j]; // Values below the diagonal from input2
                    }
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int n = dOutput.Rows;
            Matrix dMatrix1 = new Matrix(n, n);
            Matrix dMatrix2 = new Matrix(n, n);

            Parallel.For(0, n, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j)
                    {
                        // Diagonal elements do not contribute to the gradients
                        dMatrix1[i][j] = 0;
                        dMatrix2[i][j] = 0;
                    }
                    else if (i < j)
                    {
                        // Gradients for the upper triangular part go to input1
                        dMatrix1[i][j] = dOutput[i][j];
                        dMatrix2[i][j] = 0; // No contribution to input2 for these elements
                    }
                    else
                    {
                        // Gradients for the lower triangular part go to input2
                        dMatrix2[i][j] = dOutput[i][j];
                        dMatrix1[i][j] = 0; // No contribution to input1 for these elements
                    }
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dMatrix1)
                .AddInputGradient(dMatrix2)
                .Build();
        }
    }
}

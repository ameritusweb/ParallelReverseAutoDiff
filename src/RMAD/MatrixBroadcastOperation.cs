//------------------------------------------------------------------------------
// <copyright file="MatrixBroadcastOperation.cs" author="ameritusweb" date="6/16/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Matrix broadcast operation.
    /// </summary>
    public class MatrixBroadcastOperation : Operation
    {
        private Matrix input;
        private int targetRows;
        private int targetCols;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixBroadcastOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateObjectArrays.AddOrUpdate(id, new[] { (object)this.input, (object)this.targetRows, (object)this.targetCols }, (_, _) => new[] { (object)this.input, (object)this.targetRows, (object)this.targetCols });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateObjectArrays[id];
            this.input = (Matrix)restored[0];
            this.targetRows = (int)restored[1];
            this.targetCols = (int)restored[2];
        }

        /// <summary>
        /// Performs the forward operation for the matrix broadcast function.
        /// </summary>
        /// <param name="input">A matrix to broadcast.</param>
        /// <param name="targetRows">The target number of rows.</param>
        /// <param name="targetCols">The target number of columns.</param>
        /// <returns>The output of the matrix broadcast operation.</returns>
        public Matrix Forward(Matrix input, int targetRows, int targetCols)
        {
            this.input = input;
            this.targetRows = targetRows;
            this.targetCols = targetCols;

            this.Output = new Matrix(targetRows, targetCols);

            Parallel.For(0, targetRows, i =>
            {
                for (int j = 0; j < targetCols; j++)
                {
                    this.Output[i][j] = input[i % input.Rows][j % input.Cols];
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput = new Matrix(this.input.Rows, this.input.Cols);

            Parallel.For(0, this.input.Rows, i =>
            {
                for (int j = 0; j < this.input.Cols; j++)
                {
                    for (int k = i; k < this.targetRows; k += this.input.Rows)
                    {
                        for (int l = j; l < this.targetCols; l += this.input.Cols)
                        {
                            dInput[i][j] += dOutput[k][l];
                        }
                    }
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .Build();
        }
    }
}

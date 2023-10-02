//------------------------------------------------------------------------------
// <copyright file="HadamardProductOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise multiplication operation.
    /// </summary>
    public class HadamardProductOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new HadamardProductOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.input1, this.input2 }, (_, _) => new[] { this.input1, this.input2 });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.input1 = restored[0];
            this.input2 = restored[1];
        }

        /// <summary>
        /// Performs the forward operation for the Hadamard product function.
        /// </summary>
        /// <param name="input1">The first input to the Hadamard product operation.</param>
        /// <param name="input2">The second input to the Hadamard product operation.</param>
        /// <returns>The output of the Hadamard product operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2)
        {
            this.input1 = input1;
            this.input2 = input2;
            int numRows = input1.Length;
            int numCols = input1[0].Length;

            this.Output = new Matrix(numRows, numCols);

            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    this.Output[i][j] = input1[i][j] * input2[i][j];
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int numRows = this.input1.Length;
            int numCols = this.input1[0].Length;

            // Calculate gradient w.r.t. input1
            Matrix dInput1 = new Matrix(numRows, numCols);

            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    dInput1[i][j] = dOutput[i][j] * this.input2[i][j];
                }
            });

            // Calculate gradient w.r.t. input2
            Matrix dInput2 = new Matrix(numRows, numCols);

            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                for (int j = 0; j < numCols; j++)
                {
                    dInput2[i][j] = dOutput[i][j] * this.input1[i][j];
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .Build();
        }
    }
}

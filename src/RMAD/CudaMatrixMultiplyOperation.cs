﻿//------------------------------------------------------------------------------
// <copyright file="CudaMatrixMultiplyOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading;
    using System.Threading.Tasks;
    using ParallelReverseAutoDiff.Exceptions;

    /// <summary>
    /// CUDA Matrix multiplication operation.
    /// </summary>
    public class CudaMatrixMultiplyOperation : Operation
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
            return new CudaMatrixMultiplyOperation();
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
        /// Performs the forward operation for the matrix multiply function.
        /// </summary>
        /// <param name="input1">The first input to the matrix multiply operation.</param>
        /// <param name="input2">The second input to the matrix multiply operation.</param>
        /// <returns>The output of the matrix multiply operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2)
        {
            if (!CudaBlas.Instance.IsInitialized)
            {
                throw new CudaNotInitializedException();
            }

            this.input1 = input1;
            this.input2 = input2;
            int input1Cols = input1[0].Length;
            int input2Rows = input2.Length;

            if (input1Cols != input2Rows)
            {
                throw new InvalidOperationException("Input 1 columns do not match Input 2 rows");
            }

            this.Output = CudaBlas.Instance.WriteMatricesToSharedMemory(input1, false, input2, false);

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            if (!CudaBlas.Instance.IsInitialized)
            {
                throw new CudaNotInitializedException();
            }

            // Calculate gradient w.r.t. input1

            // Compute dInput1 using MatrixMultiply
            Matrix? dInput1 = CudaBlas.Instance.WriteMatricesToSharedMemory(dOutput, false, this.input2, true);

            // Calculate gradient w.r.t. input2

            // Compute dInput2 using MatrixMultiply
            Matrix? dInput2 = CudaBlas.Instance.WriteMatricesToSharedMemory(this.input1, true, dOutput, false);

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .Build();
        }
    }
}

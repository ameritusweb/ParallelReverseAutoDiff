//------------------------------------------------------------------------------
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

            var context = SynchronizationContext.Current;
            context.Send(
                (_) =>
            {
                this.Output = CudaBlas.Instance.MatrixMultiply(input1, false, input2, false);
            }, null);

            return this.Output;
        }

        /// <inheritdoc />
        public override (Matrix?, Matrix?) Backward(Matrix dOutput)
        {
            if (!CudaBlas.Instance.IsInitialized)
            {
                throw new CudaNotInitializedException();
            }

            Matrix? dInput1 = null;
            Matrix? dInput2 = null;

            // Get the current synchronization context
            var context = SynchronizationContext.Current;
            context.Send(
                (_) =>
            {
                // Calculate gradient w.r.t. input1

                // Compute dInput1 using MatrixMultiply
                dInput1 = CudaBlas.Instance.MatrixMultiply(dOutput, false, this.input2, true);

                // Calculate gradient w.r.t. input2

                // Compute dInput2 using MatrixMultiply
                dInput2 = CudaBlas.Instance.MatrixMultiply(this.input1, true, dOutput, false);
            }, null);

            return (dInput1, dInput2);
        }
    }
}

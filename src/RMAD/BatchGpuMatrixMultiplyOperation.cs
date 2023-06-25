//------------------------------------------------------------------------------
// <copyright file="BatchGpuMatrixMultiplyOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;
    using ParallelReverseAutoDiff.Exceptions;

    /// <summary>
    /// Batch GPU Matrix multiplication operation.
    /// </summary>
    public class BatchGpuMatrixMultiplyOperation : BatchOperation
    {
        private GpuMatrixMultiplyOperation[] operations;

        /// <summary>
        /// Initializes a new instance of the <see cref="BatchGpuMatrixMultiplyOperation"/> class.
        /// </summary>
        /// <param name="batchSize">The batch size.</param>
        private BatchGpuMatrixMultiplyOperation(int batchSize)
        {
            this.operations = new GpuMatrixMultiplyOperation[batchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchGpuMatrixMultiplyOperation(net.Parameters.BatchSize);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.operations, (key, oldValue) => this.operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.operations = this.IntermediateOperationArrays[id].OfType<GpuMatrixMultiplyOperation>().ToArray();
        }

        /// <summary>
        /// Performs the forward operation for the matrix multiply function.
        /// </summary>
        /// <param name="input1">The first input to the matrix multiply operation.</param>
        /// <param name="input2">The second input to the matrix multiply operation.</param>
        /// <returns>The output of the matrix multiply operation.</returns>
        public DeepMatrix Forward(DeepMatrix input1, DeepMatrix input2)
        {
            if (!CudaBlas.Instance.IsInitialized)
            {
                throw new CudaNotInitializedException();
            }

            var matrixArray = new Matrix[input1.Depth];
            Parallel.For(0, input2.Depth, i =>
            {
                this.operations[i] = new GpuMatrixMultiplyOperation();
                matrixArray[i] = this.operations[i].Forward(input1[i], input2[i]);
            });

            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <summary>
        /// Performs the forward operation for the matrix multiply function.
        /// </summary>
        /// <param name="input1">The first input to the matrix multiply operation.</param>
        /// <param name="input2">The second input to the matrix multiply operation.</param>
        /// <returns>The output of the matrix multiply operation.</returns>
        public DeepMatrix Forward(DeepMatrix input1, Matrix input2)
        {
            if (!CudaBlas.Instance.IsInitialized)
            {
                throw new CudaNotInitializedException();
            }

            var matrixArray = new Matrix[input1.Depth];
            Parallel.For(0, input1.Depth, i =>
            {
                this.operations[i] = new GpuMatrixMultiplyOperation();
                matrixArray[i] = this.operations[i].Forward(input1[i], input2);
            });

            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <summary>
        /// Performs the forward operation for the matrix multiply function.
        /// </summary>
        /// <param name="input1">The first input to the matrix multiply operation.</param>
        /// <param name="input2">The second input to the matrix multiply operation.</param>
        /// <returns>The output of the matrix multiply operation.</returns>
        public DeepMatrix Forward(Matrix input1, DeepMatrix input2)
        {
            if (!CudaBlas.Instance.IsInitialized)
            {
                throw new CudaNotInitializedException();
            }

            var matrixArray = new Matrix[input2.Depth];
            Parallel.For(0, input2.Depth, i =>
            {
                this.operations[i] = new GpuMatrixMultiplyOperation();
                matrixArray[i] = this.operations[i].Forward(input1, input2[i]);
            });

            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override BackwardResult[] Backward(DeepMatrix dOutput)
        {
            if (!CudaBlas.Instance.IsInitialized)
            {
                throw new CudaNotInitializedException();
            }

            var result = new BackwardResult[dOutput.Depth];
            Parallel.For(0, dOutput.Depth, i =>
            {
                result[i] = this.operations[i].Backward(dOutput[i]);
            });
            return result;
        }
    }
}

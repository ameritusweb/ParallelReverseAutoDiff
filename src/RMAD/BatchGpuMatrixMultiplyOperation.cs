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
    public class BatchGpuMatrixMultiplyOperation : BatchOperation<GpuMatrixMultiplyOperation>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BatchGpuMatrixMultiplyOperation"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        private BatchGpuMatrixMultiplyOperation(NeuralNetwork net)
            : base(net)
        {
            this.Operations = new GpuMatrixMultiplyOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchGpuMatrixMultiplyOperation(net);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.Operations, (x, y) => this.Operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.Operations = this.IntermediateOperationArrays[id].OfType<GpuMatrixMultiplyOperation>().ToArray();
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

            this.ExtendOperations();

            var matrixArray = new Matrix[input1.Depth];
            Parallel.For(0, input2.Depth, i =>
            {
                this.Operations[i] = new GpuMatrixMultiplyOperation();
                matrixArray[i] = this.Operations[i].Forward(input1[i], input2[i]);
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

            this.ExtendOperations();

            var matrixArray = new Matrix[input1.Depth];
            Parallel.For(0, input1.Depth, i =>
            {
                this.Operations[i] = new GpuMatrixMultiplyOperation();
                matrixArray[i] = this.Operations[i].Forward(input1[i], input2);
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

            this.ExtendOperations();

            var matrixArray = new Matrix[input2.Depth];
            Parallel.For(0, input2.Depth, i =>
            {
                this.Operations[i] = new GpuMatrixMultiplyOperation();
                matrixArray[i] = this.Operations[i].Forward(input1, input2[i]);
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
                result[i] = this.Operations[i].Backward(dOutput[i]);
            });
            return result;
        }
    }
}

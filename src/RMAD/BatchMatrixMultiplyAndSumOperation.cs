//------------------------------------------------------------------------------
// <copyright file="BatchMatrixMultiplyAndSumOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch matrix multiply and sum operation.
    /// </summary>
    public class BatchMatrixMultiplyAndSumOperation : BatchOperation<MatrixMultiplyAndSumOperation>
    {
        private BatchMatrixMultiplyAndSumOperation(NeuralNetwork net)
            : base(net)
        {
            this.Operations = new MatrixMultiplyAndSumOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchMatrixMultiplyAndSumOperation(net);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.Operations, (key, oldValue) => this.Operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.Operations = this.IntermediateOperationArrays[id].OfType<MatrixMultiplyAndSumOperation>().ToArray();
        }

        /// <summary>
        /// Performs the forward operation for the Matrix Multiply and Sum function.
        /// </summary>
        /// <param name="input1">The first input to the Matrix Multiply and Sum operation.</param>
        /// <param name="input2">The second input to the Matrix Multiply and Sum operation.</param>
        /// <returns>The output of the Matrix Multiply and Sum operation.</returns>
        public DeepMatrix Forward(DeepMatrix input1, DeepMatrix input2)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input1.Depth];
            for (int i = 0; i < input1.Depth; i++)
            {
                this.Operations[i] = new MatrixMultiplyAndSumOperation();
                matrixArray[i] = this.Operations[i].Forward(input1[i], input2);
            }

            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override BackwardResult[] Backward(DeepMatrix dOutput)
        {
            var result = new BackwardResult[dOutput.Depth];
            Parallel.For(0, dOutput.Depth, i =>
            {
                result[i] = this.Operations[i].Backward(dOutput[i]);
            });
            return result;
        }
    }
}

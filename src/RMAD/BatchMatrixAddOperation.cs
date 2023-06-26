//------------------------------------------------------------------------------
// <copyright file="BatchMatrixAddOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Threading.Tasks;

    /// <summary>
    /// Batch matrix addition operation.
    /// </summary>
    public class BatchMatrixAddOperation : BatchOperation<MatrixAddOperation>
    {
        private MatrixAddOperation[] operations;

        private BatchMatrixAddOperation(NeuralNetwork net)
            : base(net)
        {
            this.operations = new MatrixAddOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchMatrixAddOperation(net);
        }

        /// <summary>
        /// Performs the forward operation for the matrix add function.
        /// </summary>
        /// <param name="inputA">The first input to the matrix add operation.</param>
        /// <param name="inputB">The second input to the matrix add operation.</param>
        /// <returns>The output of the matrix add operation.</returns>
        public DeepMatrix Forward(DeepMatrix inputA, DeepMatrix inputB)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[inputA.Depth];
            Parallel.For(0, inputA.Depth, i =>
            {
                this.operations[i] = new MatrixAddOperation();
                matrixArray[i] = this.operations[i].Forward(inputA[i], inputB[i]);
            });
            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <summary>
        /// Performs the forward operation for the matrix add function.
        /// </summary>
        /// <param name="inputA">The first input to the matrix add operation.</param>
        /// <param name="inputB">The second input to the matrix add operation.</param>
        /// <returns>The output of the matrix add operation.</returns>
        public DeepMatrix Forward(DeepMatrix inputA, Matrix inputB)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[inputA.Depth];
            Parallel.For(0, inputA.Depth, i =>
            {
                this.operations[i] = new MatrixAddOperation();
                matrixArray[i] = this.operations[i].Forward(inputA[i], inputB);
            });
            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override BackwardResult[] Backward(DeepMatrix dOutput)
        {
            var result = new BackwardResult[dOutput.Depth];
            Parallel.For(0, dOutput.Depth, i =>
            {
                result[i] = this.operations[i].Backward(dOutput[i]);
            });
            return result;
        }
    }
}

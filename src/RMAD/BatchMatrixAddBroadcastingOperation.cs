//------------------------------------------------------------------------------
// <copyright file="BatchMatrixAddBroadcastingOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Threading.Tasks;

    /// <summary>
    /// Batch matrix addition broadcasting operation.
    /// </summary>
    public class BatchMatrixAddBroadcastingOperation : BatchOperation<MatrixAddBroadcastingOperation>
    {
        private BatchMatrixAddBroadcastingOperation(NeuralNetwork net)
            : base(net)
        {
            this.Operations = new MatrixAddBroadcastingOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchMatrixAddBroadcastingOperation(net);
        }

        /// <summary>
        /// Performs the forward operation for the matrix add broadcasting function.
        /// </summary>
        /// <param name="input">The first input to the matrix add broadcasting operation.</param>
        /// <param name="bias">The bias to the matrix add broadcasting operation, a 1xN matrix where N is the number of input columns.</param>
        /// <returns>The output of the matrix add boradcasting operation.</returns>
        public DeepMatrix Forward(DeepMatrix input, Matrix bias)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.Operations[i] = new MatrixAddBroadcastingOperation();
                matrixArray[i] = this.Operations[i].Forward(input[i], bias);
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
                result[i] = this.Operations[i].Backward(dOutput[i]);
            });
            return result;
        }
    }
}

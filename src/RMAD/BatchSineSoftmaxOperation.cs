//------------------------------------------------------------------------------
// <copyright file="BatchSineSoftmaxOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Batch sine-softmax operation.
    /// </summary>
    public class BatchSineSoftmaxOperation : BatchOperation<SineSoftmaxOperation>
    {
        private BatchSineSoftmaxOperation(NeuralNetwork net)
            : base(net)
        {
            this.Operations = new SineSoftmaxOperation[net.Parameters.BatchSize];
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IBatchOperation Instantiate(NeuralNetwork net)
        {
            return new BatchSineSoftmaxOperation(net);
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateOperationArrays.AddOrUpdate(id, this.Operations, (x, y) => this.Operations);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.Operations = this.IntermediateOperationArrays[id].OfType<SineSoftmaxOperation>().ToArray();
        }

        /// <summary>
        /// Performs the forward operation for the sine-softmax function.
        /// </summary>
        /// <param name="input">The input to the sine-softmax operation.</param>
        /// <returns>The output of the sine-softmax operation.</returns>
        public DeepMatrix Forward(DeepMatrix input)
        {
            this.ExtendOperations();
            var matrixArray = new Matrix[input.Depth];
            Parallel.For(0, input.Depth, i =>
            {
                this.Operations[i] = new SineSoftmaxOperation();
                matrixArray[i] = this.Operations[i].Forward(input[i]);
            });
            this.DeepOutput = new DeepMatrix(matrixArray);
            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override BackwardResult[] Backward(DeepMatrix dLdOutput)
        {
            var result = new BackwardResult[dLdOutput.Depth];
            Parallel.For(0, dLdOutput.Depth, i =>
            {
                result[i] = this.Operations[i].Backward(dLdOutput[i]);
            });
            return result;
        }
    }
}

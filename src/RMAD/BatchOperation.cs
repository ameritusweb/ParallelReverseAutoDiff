//------------------------------------------------------------------------------
// <copyright file="BatchOperation.cs" author="ameritusweb" date="6/24/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <inheritdoc />
    public abstract class BatchOperation<T> : OperationBase, IBatchOperation
        where T : IOperationBase
    {
        private T[] operations;
        private NeuralNetwork network;

        /// <summary>
        /// Initializes a new instance of the <see cref="BatchOperation{T}"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public BatchOperation(NeuralNetwork net)
        {
            this.operations = new T[net.Parameters.BatchSize];
            this.network = net;
        }

        /// <summary>
        /// Gets the array of operations.
        /// </summary>
        public T[] Operations
        {
            get
            {
                return this.operations;
            }
        }

        /// <summary>
        /// Abstract method to perform backward pass, must be implemented by derived classes.
        /// </summary>
        /// <param name="dOutput">The upstream gradient.</param>
        /// <returns>The gradients to send to the adjacent backward operations.</returns>
        public abstract BackwardResult[] Backward(DeepMatrix dOutput);

        /// <summary>
        /// Extends the operations by the current batch size.
        /// </summary>
        protected virtual void ExtendOperations()
        {
            int initialSize = this.operations.Length;
            if (initialSize != this.network.Parameters.BatchSize)
            {
                Array.Resize(array: ref this.operations, newSize: this.network.Parameters.BatchSize);
            }
        }
    }
}

//------------------------------------------------------------------------------
// <copyright file="ConvolutionalComputationGraph.cs" author="ameritusweb" date="5/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.CnnExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A convolutional computation graph.
    /// </summary>
    public class ConvolutionalComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ConvolutionalComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public ConvolutionalComputationGraph(ConvolutionalNeuralNetwork net)
            : base(net)
        {
        }

        /// <summary>
        /// Lifecycle method to setup the dependencies of the computation graph.
        /// </summary>
        /// <param name="operation">The operation.</param>
        protected override void DependenciesSetup(IOperationBase operation)
        {
            base.DependenciesSetup(operation);
        }
    }
}

//------------------------------------------------------------------------------
// <copyright file="NestedFeedForwardComputationGraph.cs" author="ameritusweb" date="5/9/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FeedForwardExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A feed forward computation graph.
    /// </summary>
    public class NestedFeedForwardComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="NestedFeedForwardComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public NestedFeedForwardComputationGraph(NestedFeedForwardNeuralNetwork net)
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

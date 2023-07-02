// ------------------------------------------------------------------------------
// <copyright file="AttentionMessagePassingComputationGraph.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.AttentionMessagePassing
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An attention message passing computation graph.
    /// </summary>
    public class AttentionMessagePassingComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AttentionMessagePassingComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public AttentionMessagePassingComputationGraph(AttentionMessagePassingNeuralNetwork net)
            : base(net)
        {
        }

        /// <summary>
        /// Lifecycle method to setup the dependencies of the computation graph.
        /// </summary>
        /// <param name="operation">The operation.</param>
        /// <param name="layerInfo">The layer information.</param>
        protected override void DependenciesSetup(IOperationBase operation, LayerInfo layerInfo)
        {
            base.DependenciesSetup(operation, layerInfo);
        }
    }
}

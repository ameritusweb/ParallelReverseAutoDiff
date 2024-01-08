// ------------------------------------------------------------------------------
// <copyright file="GraphAttentionComputationGraph.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GatExample.OpticalCharacterRecognition2.GraphAttentionNetwork
{
    using ParallelReverseAutoDiff.RMAD;

    public class GraphAttentionComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GraphAttentionComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public GraphAttentionComputationGraph(GraphAttentionNetwork net)
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

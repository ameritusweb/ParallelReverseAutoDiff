// ------------------------------------------------------------------------------
// <copyright file="GravitationalComputationGraph.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GravNetExample.GravitationalNetwork
{
    using ParallelReverseAutoDiff.RMAD;

    public class GravitationalComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GravitationalComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public GravitationalComputationGraph(GravitationalNetwork net)
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

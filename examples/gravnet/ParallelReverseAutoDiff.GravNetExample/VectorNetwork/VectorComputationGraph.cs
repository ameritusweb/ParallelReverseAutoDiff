// ------------------------------------------------------------------------------
// <copyright file="VectorComputationGraph.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GravNetExample.VectorNetwork
{
    using ParallelReverseAutoDiff.RMAD;

    public class VectorComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="VectorComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public VectorComputationGraph(VectorNetwork net)
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

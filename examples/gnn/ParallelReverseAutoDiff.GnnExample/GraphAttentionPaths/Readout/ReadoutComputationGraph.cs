﻿// ------------------------------------------------------------------------------
// <copyright file="ReadoutComputationGraph.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A readout computation graph.
    /// </summary>
    public class ReadoutComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ReadoutComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public ReadoutComputationGraph(ReadoutNeuralNetwork net)
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

﻿// ------------------------------------------------------------------------------
// <copyright file="LstmComputationGraph.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionLstm.LSTM
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An LSTM computation graph.
    /// </summary>
    public class LstmComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LstmComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public LstmComputationGraph(LstmNeuralNetwork net)
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

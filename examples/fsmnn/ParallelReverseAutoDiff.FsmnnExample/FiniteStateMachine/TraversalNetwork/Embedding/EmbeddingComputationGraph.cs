// ------------------------------------------------------------------------------
// <copyright file="EmbeddingComputationGraph.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.TraversalNetwork.Embedding
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An embedding computation graph.
    /// </summary>
    public class EmbeddingComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EmbeddingComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public EmbeddingComputationGraph(EmbeddingNeuralNetwork net)
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

        /// <summary>
        /// Lifecycle method to retrieve the type of the operation.
        /// </summary>
        /// <param name="type">The type.</param>
        /// <returns>The retrieved type.</returns>
        protected override Type TypeRetrieved(string type)
        {
            var typeRetrieved = base.TypeRetrieved(type);
            return typeRetrieved;
        }
    }
}

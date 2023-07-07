// ------------------------------------------------------------------------------
// <copyright file="TransformerComputationGraph.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.Transformer
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A transformer computation graph.
    /// </summary>
    public class TransformerComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TransformerComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public TransformerComputationGraph(TransformerNeuralNetwork net)
            : base(net)
        {
        }

        /// <summary>
        /// Lifecycle method to setup the dependencies of the computation graph.
        /// </summary>
        /// <param name="operation">The operation.</param>
        protected override void DependenciesSetup(IOperationBase operation, LayerInfo layerInfo)
        {
            base.DependenciesSetup(operation, layerInfo);
        }
    }
}

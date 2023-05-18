//------------------------------------------------------------------------------
// <copyright file="SelfAttentionMultiLayerLSTMComputationGraph.cs" author="ameritusweb" date="5/10/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.LstmExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The computation graph for a self-attention multi-layer LSTM.
    /// </summary>
    public class SelfAttentionMultiLayerLSTMComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SelfAttentionMultiLayerLSTMComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public SelfAttentionMultiLayerLSTMComputationGraph(SelfAttentionMultiLayerLSTM net)
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

namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.EdgeAttention
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An edge attention computation graph.
    /// </summary>
    public class EdgeAttentionComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EdgeAttentionComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public EdgeAttentionComputationGraph(EdgeAttentionNeuralNetwork net)
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

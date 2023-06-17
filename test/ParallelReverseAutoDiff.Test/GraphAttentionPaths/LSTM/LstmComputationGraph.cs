namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN
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
        protected override void DependenciesSetup(IOperationBase operation)
        {
            base.DependenciesSetup(operation);
        }
    }
}

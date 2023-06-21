using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.Convolutional
{
    /// <summary>
    /// A convolutional computation graph.
    /// </summary>
    public class ConvolutionalComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ConvolutionalComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public ConvolutionalComputationGraph(ConvolutionalNeuralNetwork net)
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

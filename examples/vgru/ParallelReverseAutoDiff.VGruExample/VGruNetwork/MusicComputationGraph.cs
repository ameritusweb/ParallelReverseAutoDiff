// ------------------------------------------------------------------------------
// <copyright file="MusicComputationGraph.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample.VGruNetwork
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A gated recurrent computation graph.
    /// </summary>
    public class MusicComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MusicComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public MusicComputationGraph(MusicNetwork net)
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
        /// Type retrieval method.
        /// </summary>
        /// <param name="type">The type.</param>
        /// <returns>The retrieved type.</returns>
        protected override Type TypeRetrieved(string type)
        {
            var aa = base.TypeRetrieved(type);
            var actualType = Type.GetType("ParallelReverseAutoDiff.RMAD." + type);
            if (actualType == null)
            {
                return aa;
            }
            else
            {
                return actualType;
            }
        }
    }
}

// ------------------------------------------------------------------------------
// <copyright file="VectorFieldComputationGraph.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.VGruExample.VGruNetwork
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A vector field computation graph.
    /// </summary>
    public class VectorFieldComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="VectorFieldComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public VectorFieldComputationGraph(VectorFieldNetwork net)
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
        /// Gets the type.
        /// </summary>
        /// <param name="type">The type.</param>
        /// <returns>Returns the type.</returns>
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

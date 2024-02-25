// ------------------------------------------------------------------------------
// <copyright file="GlyphComputationGraph.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GravNetExample.GlyphNetwork
{
    using ParallelReverseAutoDiff.RMAD;
    using System;

    public class GlyphComputationGraph : ComputationGraph
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GlyphComputationGraph"/> class.
        /// </summary>
        /// <param name="net">The neural network.</param>
        public GlyphComputationGraph(GlyphNetwork net)
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

        protected override Type TypeRetrieved(string type)
        {
            var aa = base.TypeRetrieved(type);
            var actualType = Type.GetType("ParallelReverseAutoDiff.RMAD." + type);
            if (actualType == null)
            {
                return aa;
            } else
            {
                return actualType;
            }
        }
    }
}

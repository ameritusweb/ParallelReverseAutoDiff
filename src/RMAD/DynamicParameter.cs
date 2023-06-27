//------------------------------------------------------------------------------
// <copyright file="DynamicParameter.cs" author="ameritusweb" date="6/27/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Provides a dynamic parameter.
    /// </summary>
    public class DynamicParameter
    {
        private readonly ComputationGraph graph;
        private readonly string inputName;
        private readonly LayerInfo layerInfo;

        /// <summary>
        /// Initializes a new instance of the <see cref="DynamicParameter"/> class.
        /// </summary>
        /// <param name="graph">The computational graph.</param>
        /// <param name="inputName">The input name.</param>
        /// <param name="layerInfo">The layer info.</param>
        public DynamicParameter(ComputationGraph graph, string inputName, LayerInfo layerInfo)
        {
            this.graph = graph;
            this.inputName = inputName;
            this.layerInfo = layerInfo;
        }

        /// <summary>
        /// Get the value of the dynamic parameter.
        /// </summary>
        /// <returns>The value.</returns>
        public object GetValue()
        {
            return this.graph[MatrixType.Dynamic, this.inputName, this.layerInfo];
        }

        /// <summary>
        /// Get the value of the dynamic parameter.
        /// </summary>
        /// <param name="graph">The computation graph.</param>
        /// <returns>The value.</returns>
        public object GetValue(ComputationGraph graph)
        {
            return graph[MatrixType.Dynamic, this.inputName, this.layerInfo];
        }
    }
}

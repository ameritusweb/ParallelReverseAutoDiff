//------------------------------------------------------------------------------
// <copyright file="ComputationGraph.cs" author="ameritusweb" date="5/8/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using ParallelReverseAutoDiff.Interprocess;

    /// <summary>
    /// A computation graph.
    /// </summary>
    public abstract class ComputationGraph
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.RMAD";
        private readonly ConcurrentDictionary<string, Func<LayerInfo, object>> weightsAndBiases = new ConcurrentDictionary<string, Func<LayerInfo, object>>();
        private readonly ConcurrentDictionary<string, Func<LayerInfo, object>> gradients = new ConcurrentDictionary<string, Func<LayerInfo, object>>();
        private readonly ConcurrentDictionary<string, Func<LayerInfo, object>> intermediates = new ConcurrentDictionary<string, Func<LayerInfo, object>>();
        private readonly ConcurrentDictionary<string, Func<LayerInfo, object>> scalars = new ConcurrentDictionary<string, Func<LayerInfo, object>>();
        private readonly ConcurrentDictionary<string, Func<LayerInfo, object>> operationFinders = new ConcurrentDictionary<string, Func<LayerInfo, object>>();
        private readonly ConcurrentDictionary<string, IOperationBase> operations = new ConcurrentDictionary<string, IOperationBase>();
        private readonly ConcurrentHashSet<string> nestedOperations = new ConcurrentHashSet<string>();
        private readonly NeuralNetwork neuralNetwork;
        private IOperationBase? startOperation;
        private IOperationBase? currentOperation;

        /// <summary>
        /// Initializes a new instance of the <see cref="ComputationGraph"/> class.
        /// </summary>
        /// <param name="neuralNetwork">The neural network.</param>
        protected ComputationGraph(NeuralNetwork neuralNetwork)
        {
            this.neuralNetwork = neuralNetwork;
        }

        /// <summary>
        /// Gets the start operation.
        /// </summary>
        public IOperationBase? StartOperation
        {
            get
            {
                return this.startOperation;
            }
        }

        /// <summary>
        /// Gets the current operation.
        /// </summary>
        public IOperationBase? CurrentOperation
        {
            get
            {
                return this.currentOperation;
            }
        }

        /// <summary>
        /// Retrieve the operation by the operation identifier.
        /// </summary>
        /// <param name="operationIdentifier">The operation identifier.</param>
        /// <returns>The operation.</returns>
        public IOperationBase this[string operationIdentifier]
        {
            get
            {
                return this.operations[operationIdentifier];
            }
        }

        /// <summary>
        /// Retrieve the weight or gradient matrix by type and identifier.
        /// </summary>
        /// <param name="type">The type of matrix.</param>
        /// <param name="identifier">The identifier.</param>
        /// <param name="index">The matrix index.</param>
        /// <returns>The weight or gradient matrix.</returns>
        public object this[MatrixType type, string identifier, LayerInfo index]
        {
            get
            {
                return type switch
                {
                    MatrixType.Weight => this.weightsAndBiases[identifier](index),
                    MatrixType.Bias => this.weightsAndBiases[identifier](index),
                    MatrixType.Gradient => this.gradients[identifier](index),
                    MatrixType.Intermediate => this.intermediates[identifier](index),
                    _ => throw new ArgumentException("Invalid matrix type."),
                };
            }
        }

        /// <summary>
        /// Retrieve an array of operations starting from the start position inclusive, and ending at the end position inclusive.
        /// </summary>
        /// <param name="identifier">The identifier not including the time step or layer.</param>
        /// <param name="start">The start position.</param>
        /// <param name="end">The end position.</param>
        /// <returns>The array of operations.</returns>
        public IOperationBase[] ToOperationArray(string identifier, LayerInfo start, LayerInfo end)
        {
            List<IOperationBase> operationList = new List<IOperationBase>();

            for (int t = start.TimeStep; t <= end.TimeStep; ++t)
            {
                int startLayer = t == start.TimeStep ? start.Layer : 0;
                int l = startLayer;
                while (true)
                {
                    // If not nested, directly check operation with layer
                    if (start.Type != LayerInfoType.Nested)
                    {
                        var key = $"{identifier}_{t}_{l}";
                        if (!this.operations.ContainsKey(key))
                        {
                            break;  // Break the loop if no operation for the current layer index.
                        }

                        operationList.Add(this.operations[key]);
                        l++;
                        continue;  // Continue the loop if operation found
                    }

                    // If it is nested, check operations with nested layers
                    int startNestedLayer = t == start.TimeStep && l == startLayer ? start.NestedLayer : 0;
                    int nl = startNestedLayer;
                    while (true)
                    {
                        var key = $"{identifier}_{t}_{l}_{nl}";
                        if (!this.operations.ContainsKey(key))
                        {
                            break;  // Break the inner loop if no operation for the current nested layer index.
                        }

                        operationList.Add(this.operations[key]);
                        nl++;
                    }

                    // If the nested layer index didn't increase, there's no operation for the current layer index.
                    if (nl == startNestedLayer)
                    {
                        break;  // Break the outer loop
                    }

                    l++;
                }
            }

            return operationList.ToArray();
        }

        /// <summary>
        /// Stores the operation intermediates.
        /// </summary>
        /// <param name="id">The ID.</param>
        public void StoreOperationIntermediates(Guid id)
        {
            var operations = this.operations.Values;
            foreach (var operation in operations)
            {
                operation.Store(id);
            }
        }

        /// <summary>
        /// Restores the operation intermediates.
        /// </summary>
        /// <param name="id">The ID.</param>
        public void RestoreOperationIntermediates(Guid id)
        {
            var operations = this.operations.Values;
            foreach (var operation in operations)
            {
                operation.Restore(id);
            }
        }

        /// <summary>
        /// Construct the computation graph from an architecture with no layers and no temporal component.
        /// </summary>
        /// <param name="architecture">The architecture.</param>
        /// <returns>The computation graph.</returns>
        public ComputationGraph ConstructFromArchitecture(JsonArchitecture architecture)
        {
            LayerInfo layerInfo = LayerInfo.Empty;
            foreach (var timeStep in architecture.TimeSteps)
            {
                if (timeStep.StartOperations != null)
                {
                    foreach (var operationInfo in timeStep.StartOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }

                if (timeStep.EndOperations != null)
                {
                    foreach (var operationInfo in timeStep.EndOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }
            }

            return this;
        }

        /// <summary>
        /// Construct the computation graph from an architecture with layers and no temporal component.
        /// </summary>
        /// <param name="architecture">The architecture.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <returns>The computation graph.</returns>
        public ComputationGraph ConstructFromArchitecture(JsonArchitecture architecture, int numLayers)
        {
            var layerInfo = LayerInfo.Empty;
            foreach (var timeStep in architecture.TimeSteps)
            {
                if (timeStep.StartOperations != null)
                {
                    foreach (var operationInfo in timeStep.StartOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }

                for (int l = 0; l < numLayers; l++)
                {
                    layerInfo.Layer = l;
                    foreach (var layer in timeStep.Layers)
                    {
                        foreach (var operationInfo in layer.Operations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }
                }

                layerInfo.Layer = 0;

                if (timeStep.EndOperations != null)
                {
                    foreach (var operationInfo in timeStep.EndOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }
            }

            return this;
        }

        /// <summary>
        /// Construct the computation graph from an architecture with nested layers and no temporal component.
        /// </summary>
        /// <param name="architecture">The architecture.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numNestedLayers">The number of nested layers.</param>
        /// <returns>The computation graph.</returns>
        public ComputationGraph ConstructFromArchitecture(NestedLayersJsonArchitecture architecture, int numLayers, int numNestedLayers)
        {
            var layerInfo = LayerInfo.Empty;
            foreach (var timeStep in architecture.TimeSteps)
            {
                if (timeStep.StartOperations != null)
                {
                    foreach (var operationInfo in timeStep.StartOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }

                for (int l = 0; l < numLayers; l++)
                {
                    layerInfo.Layer = l;
                    foreach (var layer in timeStep.Layers)
                    {
                        if (layer.StartOperations != null)
                        {
                            foreach (var operationInfo in layer.StartOperations)
                            {
                                this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                            }
                        }

                        layerInfo.Type = LayerInfoType.Nested;
                        for (int nl = 0; nl < numNestedLayers; nl++)
                        {
                            layerInfo.NestedLayer = nl;
                            foreach (var nestedLayer in layer.Layers)
                            {
                                foreach (var operationInfo in nestedLayer.Operations)
                                {
                                    this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                                }
                            }
                        }

                        layerInfo.NestedLayer = 0;
                        layerInfo.Type = LayerInfoType.Normal;

                        if (layer.EndOperations != null)
                        {
                            foreach (var operationInfo in layer.EndOperations)
                            {
                                this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                            }
                        }
                    }
                }

                layerInfo.Layer = 0;

                if (timeStep.EndOperations != null)
                {
                    foreach (var operationInfo in timeStep.EndOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }
            }

            return this;
        }

        /// <summary>
        /// Construct the computation graph from a dual layers architecture with layers and no temporal component.
        /// </summary>
        /// <param name="architecture">The architecture.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <returns>The computation graph.</returns>
        public ComputationGraph ConstructFromArchitecture(DualLayersJsonArchitecture architecture, int numLayers)
        {
            var layerInfo = LayerInfo.Empty;
            foreach (var timeStep in architecture.TimeSteps)
            {
                if (timeStep.StartOperations != null)
                {
                    foreach (var operationInfo in timeStep.StartOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }

                for (int l = 0; l < numLayers; l++)
                {
                    layerInfo.Layer = l;
                    foreach (var layer in timeStep.FirstLayers)
                    {
                        foreach (var operationInfo in layer.Operations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }
                }

                layerInfo.Layer = 0;

                if (timeStep.MiddleOperations != null)
                {
                    foreach (var operationInfo in timeStep.MiddleOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }

                for (int l = 0; l < numLayers; l++)
                {
                    layerInfo.Layer = l;
                    foreach (var layer in timeStep.SecondLayers)
                    {
                        foreach (var operationInfo in layer.Operations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }
                }

                layerInfo.Layer = 0;

                if (timeStep.EndOperations != null)
                {
                    foreach (var operationInfo in timeStep.EndOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }
            }

            return this;
        }

        /// <summary>
        /// Construct the computation graph from a triple layers architecture with layers and no temporal component.
        /// </summary>
        /// <param name="architecture">The architecture.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <returns>The computation graph.</returns>
        public ComputationGraph ConstructFromArchitecture(TripleLayersJsonArchitecture architecture, int numLayers)
        {
            var layerInfo = LayerInfo.Empty;
            foreach (var timeStep in architecture.TimeSteps)
            {
                if (timeStep.StartOperations != null)
                {
                    foreach (var operationInfo in timeStep.StartOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }

                for (int l = 0; l < numLayers; l++)
                {
                    layerInfo.Layer = l;
                    foreach (var layer in timeStep.FirstLayers)
                    {
                        foreach (var operationInfo in layer.Operations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }
                }

                layerInfo.Layer = 0;

                if (timeStep.PostFirstOperations != null)
                {
                    foreach (var operationInfo in timeStep.PostFirstOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }

                for (int l = 0; l < numLayers; l++)
                {
                    layerInfo.Layer = l;
                    foreach (var layer in timeStep.SecondLayers)
                    {
                        foreach (var operationInfo in layer.Operations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }
                }

                layerInfo.Layer = 0;

                if (timeStep.PostSecondOperations != null)
                {
                    foreach (var operationInfo in timeStep.PostSecondOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }

                for (int l = 0; l < numLayers; l++)
                {
                    layerInfo.Layer = l;
                    foreach (var layer in timeStep.ThirdLayers)
                    {
                        foreach (var operationInfo in layer.Operations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }
                }

                layerInfo.Layer = 0;

                if (timeStep.EndOperations != null)
                {
                    foreach (var operationInfo in timeStep.EndOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                    }
                }
            }

            return this;
        }

        /// <summary>
        /// Construct the computation graph from a dual layers architecture with time steps and layers.
        /// </summary>
        /// <param name="architecture">The architecture.</param>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <returns>The computation graph.</returns>
        public ComputationGraph ConstructFromArchitecture(DualLayersJsonArchitecture architecture, int numTimeSteps, int numLayers)
        {
            var layerInfo = LayerInfo.Empty;
            for (int t = 0; t < numTimeSteps; t++)
            {
                layerInfo.TimeStep = t;
                foreach (var timeStep in architecture.TimeSteps)
                {
                    if (timeStep.StartOperations != null)
                    {
                        foreach (var operationInfo in timeStep.StartOperations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }

                    for (int l = 0; l < numLayers; l++)
                    {
                        layerInfo.Layer = l;
                        foreach (var layer in timeStep.FirstLayers)
                        {
                            foreach (var operationInfo in layer.Operations)
                            {
                                this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                            }
                        }
                    }

                    layerInfo.Layer = 0;

                    if (timeStep.MiddleOperations != null)
                    {
                        foreach (var operationInfo in timeStep.MiddleOperations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }

                    for (int l = 0; l < numLayers; l++)
                    {
                        layerInfo.Layer = l;
                        foreach (var layer in timeStep.SecondLayers)
                        {
                            foreach (var operationInfo in layer.Operations)
                            {
                                this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                            }
                        }
                    }

                    layerInfo.Layer = 0;

                    if (timeStep.EndOperations != null)
                    {
                        foreach (var operationInfo in timeStep.EndOperations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }
                }
            }

            return this;
        }

        /// <summary>
        /// Construct the computation graph from a triple layers architecture with time steps and layers.
        /// </summary>
        /// <param name="architecture">The architecture.</param>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <returns>The computation graph.</returns>
        public ComputationGraph ConstructFromArchitecture(TripleLayersJsonArchitecture architecture, int numTimeSteps, int numLayers)
        {
            var layerInfo = LayerInfo.Empty;
            for (int t = 0; t < numTimeSteps; t++)
            {
                layerInfo.TimeStep = t;
                foreach (var timeStep in architecture.TimeSteps)
                {
                    if (timeStep.StartOperations != null)
                    {
                        foreach (var operationInfo in timeStep.StartOperations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }

                    for (int l = 0; l < numLayers; l++)
                    {
                        layerInfo.Layer = l;
                        foreach (var layer in timeStep.FirstLayers)
                        {
                            foreach (var operationInfo in layer.Operations)
                            {
                                this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                            }
                        }
                    }

                    layerInfo.Layer = 0;

                    if (timeStep.PostFirstOperations != null)
                    {
                        foreach (var operationInfo in timeStep.PostFirstOperations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }

                    for (int l = 0; l < numLayers; l++)
                    {
                        layerInfo.Layer = l;
                        foreach (var layer in timeStep.SecondLayers)
                        {
                            foreach (var operationInfo in layer.Operations)
                            {
                                this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                            }
                        }
                    }

                    layerInfo.Layer = 0;

                    if (timeStep.PostSecondOperations != null)
                    {
                        foreach (var operationInfo in timeStep.PostSecondOperations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }

                    for (int l = 0; l < numLayers; l++)
                    {
                        layerInfo.Layer = l;
                        foreach (var layer in timeStep.ThirdLayers)
                        {
                            foreach (var operationInfo in layer.Operations)
                            {
                                this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                            }
                        }
                    }

                    layerInfo.Layer = 0;

                    if (timeStep.EndOperations != null)
                    {
                        foreach (var operationInfo in timeStep.EndOperations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }
                }
            }

            return this;
        }

        /// <summary>
        /// Construct the computation graph from an architecture with time steps and layers.
        /// </summary>
        /// <param name="architecture">The architecture.</param>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <returns>The computation graph.</returns>
        public ComputationGraph ConstructFromArchitecture(JsonArchitecture architecture, int numTimeSteps, int numLayers)
        {
            var layerInfo = LayerInfo.Empty;
            for (int t = 0; t < numTimeSteps; t++)
            {
                layerInfo.TimeStep = t;
                foreach (var timeStep in architecture.TimeSteps)
                {
                    if (timeStep.StartOperations != null)
                    {
                        foreach (var operationInfo in timeStep.StartOperations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }

                    for (int l = 0; l < numLayers; l++)
                    {
                        layerInfo.Layer = l;
                        foreach (var layer in timeStep.Layers)
                        {
                            foreach (var operationInfo in layer.Operations)
                            {
                                this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                            }
                        }
                    }

                    layerInfo.Layer = 0;

                    if (timeStep.EndOperations != null)
                    {
                        foreach (var operationInfo in timeStep.EndOperations)
                        {
                            this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                        }
                    }
                }
            }

            return this;
        }

        /// <summary>
        /// Gets the type from a string.
        /// </summary>
        /// <param name="type">The string.</param>
        /// <returns>The type.</returns>
        public Type GetTypeFrom(string type)
        {
            return this.TypeRetrieved(type);
        }

        /// <summary>
        /// Adds a weight to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The weight.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddWeight(string identifier, Func<LayerInfo, Matrix> matrix)
        {
            this.WeightAdded(identifier, matrix);
            return this;
        }

        /// <summary>
        /// Adds a weight to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The weight.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddWeight(string identifier, Func<LayerInfo, DeepMatrix> matrix)
        {
            this.WeightAdded(identifier, matrix);
            return this;
        }

        /// <summary>
        /// Adds a weight to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The weight.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddWeight(string identifier, Func<LayerInfo, DeepMatrix[]> matrix)
        {
            this.WeightAdded(identifier, matrix);
            return this;
        }

        /// <summary>
        /// Adds a bias to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The bias.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddBias(string identifier, Func<LayerInfo, Matrix> matrix)
        {
            this.BiasAdded(identifier, matrix);
            return this;
        }

        /// <summary>
        /// Adds a gradient to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddGradient(string identifier, Func<LayerInfo, Matrix> matrix)
        {
            this.GradientAdded(identifier, matrix);
            return this;
        }

        /// <summary>
        /// Adds a gradient to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddGradient(string identifier, Func<LayerInfo, DeepMatrix> matrix)
        {
            this.GradientAdded(identifier, matrix);
            return this;
        }

        /// <summary>
        /// Adds a gradient to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddGradient(string identifier, Func<LayerInfo, DeepMatrix[]> matrix)
        {
            this.GradientAdded(identifier, matrix);
            return this;
        }

        /// <summary>
        /// Adds a gradient to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddGradient(string identifier, Func<LayerInfo, FourDimensionalMatrix> matrix)
        {
            this.GradientAdded(identifier, matrix);
            return this;
        }

        /// <summary>
        /// Adds an intermediate to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddIntermediate(string identifier, Func<LayerInfo, Matrix> matrix)
        {
            this.IntermediateAdded(identifier, matrix);
            return this;
        }

        /// <summary>
        /// Adds an intermediate to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddIntermediate(string identifier, Func<LayerInfo, DeepMatrix> matrix)
        {
            this.IntermediateAdded(identifier, matrix);
            return this;
        }

        /// <summary>
        /// Adds an intermediate to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddIntermediate(string identifier, Func<LayerInfo, DeepMatrix[]> matrix)
        {
            this.IntermediateAdded(identifier, matrix);
            return this;
        }

        /// <summary>
        /// Adds an intermediate to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddIntermediate(string identifier, Func<LayerInfo, FourDimensionalMatrix> matrix)
        {
            this.IntermediateAdded(identifier, matrix);
            return this;
        }

        /// <summary>
        /// Adds a scalar to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="scalar">The scalar.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddScalar(string identifier, Func<LayerInfo, double> scalar)
        {
            this.ScalarAdded(identifier, scalar);
            return this;
        }

        /// <summary>
        /// Adds a scalar to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="scalar">The scalar.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddScalar(string identifier, Func<LayerInfo, int> scalar)
        {
            this.ScalarAdded(identifier, scalar);
            return this;
        }

        /// <summary>
        /// Adds a operation finder to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="operationFinder">The operation finder.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddOperationFinder(string identifier, Func<LayerInfo, object> operationFinder)
        {
            this.OperationFinderAdded(identifier, operationFinder);
            return this;
        }

        /// <summary>
        /// Adds an operation to the computation graph by type.
        /// </summary>
        /// <param name="type">The type of operation.</param>
        /// <param name="info">The operation info.</param>
        /// <param name="layerInfo">The layer info.</param>
        /// <returns>The computation graph.</returns>
        public ComputationGraph AddOperationByType(Type type, OperationInfo info, LayerInfo layerInfo)
        {
            var instantiate = type.GetMethod("Instantiate");
            if (instantiate == null)
            {
                throw new InvalidOperationException($"Instantiate method should exist on operation of type {type.Name}");
            }

            IOperationBase operation = (IOperationBase)(instantiate.Invoke(null, new object[] { this.neuralNetwork }) ?? throw new Exception("Instantiate method should return a non-null operation."));
            this.AddOperation(operation, info, layerInfo);
            return this;
        }

        /// <summary>
        /// Adds an operation to the computation graph.
        /// </summary>
        /// <param name="operation">The operation to add.</param>
        /// <param name="info">The operation info.</param>
        /// <param name="layerInfo">The layer info.</param>
        /// <returns>The computation graph.</returns>
        public ComputationGraph AddOperation(IOperationBase operation, OperationInfo info, LayerInfo layerInfo)
        {
            this.OperationAdded(operation, info);
            this.InitializeOperation(operation, info, layerInfo);
            if (layerInfo.Type == LayerInfoType.Normal)
            {
                this.operations.TryAdd(operation.SpecificId, operation);
            }
            else
            {
                this.operations.TryAdd(operation.NestedSpecificId, operation);
                this.nestedOperations.Add(info.Id);
            }

            this.SetupDependencies(operation, layerInfo);
            return this;
        }

        /// <summary>
        /// Initialize the computation graph operation.
        /// </summary>
        /// <param name="operation">The operation.</param>
        /// <param name="info">The operation info.</param>
        /// <param name="layerInfo">The layer info.</param>
        protected void InitializeOperation(IOperationBase operation, OperationInfo info, LayerInfo layerInfo)
        {
            this.OperationInitialized(operation, info, layerInfo);
        }

        /// <summary>
        /// Lifecycle function for when a type is retrieved from a string.
        /// </summary>
        /// <param name="type">A string.</param>
        /// <returns>A type.</returns>
        protected virtual Type TypeRetrieved(string type)
        {
            return Type.GetType($"{NAMESPACE}.{type}");
        }

        /// <summary>
        /// Lifecycle function for when a weight is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The weight.</param>
        protected virtual void WeightAdded(string identifier, Func<LayerInfo, Matrix> matrix)
        {
            this.weightsAndBiases.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when a weight is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The weight.</param>
        protected virtual void WeightAdded(string identifier, Func<LayerInfo, DeepMatrix> matrix)
        {
            this.weightsAndBiases.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when a weight is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The weight.</param>
        protected virtual void WeightAdded(string identifier, Func<LayerInfo, DeepMatrix[]> matrix)
        {
            this.weightsAndBiases.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when a bias is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The bias.</param>
        protected virtual void BiasAdded(string identifier, Func<LayerInfo, Matrix> matrix)
        {
            this.weightsAndBiases.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when a gradient is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        protected virtual void GradientAdded(string identifier, Func<LayerInfo, Matrix> matrix)
        {
            this.gradients.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when a gradient is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        protected virtual void GradientAdded(string identifier, Func<LayerInfo, FourDimensionalMatrix> matrix)
        {
            this.gradients.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when a gradient is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        protected virtual void GradientAdded(string identifier, Func<LayerInfo, DeepMatrix> matrix)
        {
            this.gradients.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when a gradient is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        protected virtual void GradientAdded(string identifier, Func<LayerInfo, DeepMatrix[]> matrix)
        {
            this.gradients.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when an intermediate is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        protected virtual void IntermediateAdded(string identifier, Func<LayerInfo, Matrix> matrix)
        {
            this.intermediates.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when an intermediate is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        protected virtual void IntermediateAdded(string identifier, Func<LayerInfo, FourDimensionalMatrix> matrix)
        {
            this.intermediates.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when an intermediate is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        protected virtual void IntermediateAdded(string identifier, Func<LayerInfo, DeepMatrix> matrix)
        {
            this.intermediates.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when an intermediate is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        protected virtual void IntermediateAdded(string identifier, Func<LayerInfo, DeepMatrix[]> matrix)
        {
            this.intermediates.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when a scalar is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="scalar">The scalar.</param>
        protected virtual void ScalarAdded(string identifier, Func<LayerInfo, double> scalar)
        {
            Func<LayerInfo, object> scalarAsObject = li => (object)scalar(li);
            this.scalars.TryAdd(identifier, scalarAsObject);
        }

        /// <summary>
        /// Lifecycle function for when a scalar is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="scalar">The scalar.</param>
        protected virtual void ScalarAdded(string identifier, Func<LayerInfo, int> scalar)
        {
            Func<LayerInfo, object> scalarAsObject = li => (object)scalar(li);
            this.scalars.TryAdd(identifier, scalarAsObject);
        }

        /// <summary>
        /// Lifecycle function for when an operation finder is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="operationFinder">The gradient.</param>
        protected virtual void OperationFinderAdded(string identifier, Func<LayerInfo, object> operationFinder)
        {
            this.operationFinders.TryAdd(identifier, operationFinder);
        }

        /// <summary>
        /// Lifecycle function for when an operation is added to the computation graph.
        /// </summary>
        /// <param name="operation">The operation to add.</param>
        /// <param name="info">The operation info.</param>
        protected virtual void OperationAdded(IOperationBase operation, OperationInfo info)
        {
            if (this.startOperation == null)
            {
                this.startOperation = operation;
                this.currentOperation = operation;
            }
            else if (this.currentOperation != null)
            {
                this.currentOperation.Next = operation;
                this.currentOperation = operation;
            }
            else
            {
                throw new InvalidOperationException("The current operation must not be null.");
            }
        }

        /// <summary>
        /// Lifecycle function for when an operation is initialized.
        /// </summary>
        /// <param name="operation">The operation.</param>
        /// <param name="info">The operation info.</param>
        /// <param name="layerInfo">The layer info.</param>
        protected virtual void OperationInitialized(IOperationBase operation, OperationInfo info, LayerInfo layerInfo)
        {
            operation.InitializeFrom(info, this.gradients, layerInfo);
        }

        /// <summary>
        /// Setup dependencies for the operation.
        /// </summary>
        /// <param name="operation">The operation to setup.</param>
        /// <param name="layerInfo">The layer info.</param>
        protected void SetupDependencies(IOperationBase operation, LayerInfo layerInfo)
        {
            this.DependenciesSetup(operation, layerInfo);
        }

        /// <summary>
        /// Lifecycle method for when the dependencies are setup for an operation.
        /// </summary>
        /// <param name="operation">The operation.</param>
        /// <param name="layerInfo">The layer info.</param>
        protected virtual void DependenciesSetup(IOperationBase operation, LayerInfo layerInfo)
        {
            object[] parameters = new object[operation.Inputs.Count];
            for (int j = 0; j < operation.Inputs.Count; ++j)
            {
                var input = operation.Inputs[j];
                var split = input.Split(new[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                var inputName = split[0];
                if (this.nestedOperations.Contains(inputName))
                {
                    var nestedSpecificId = inputName + operation.LayerInfo.ToNestedString();
                    if (this.operations.ContainsKey(nestedSpecificId))
                    {
                        var inputOp = this.operations[nestedSpecificId];
                        inputOp.Outputs.Add(layerInfo.Type == LayerInfoType.Normal ? operation.SpecificId : operation.NestedSpecificId);
                        operation.BackwardAdjacentOperations.Add(inputOp);
                        parameters[j] = inputOp;
                        continue;
                    }
                }
                else
                {
                    var specificId = inputName + operation.LayerInfo.ToString();
                    if (this.operations.ContainsKey(specificId))
                    {
                        var inputOp = this.operations[specificId];
                        inputOp.Outputs.Add(layerInfo.Type == LayerInfoType.Normal ? operation.SpecificId : operation.NestedSpecificId);
                        operation.BackwardAdjacentOperations.Add(inputOp);
                        parameters[j] = inputOp;
                        continue;
                    }
                }

                if (this.operationFinders.ContainsKey(inputName))
                {
                    // Get the corresponding value from the dictionary using the input name
                    var finder = this.operationFinders[inputName](operation.LayerInfo);
                    if (finder is IOperationBase op)
                    {
                        op.Outputs.Add(layerInfo.Type == LayerInfoType.Normal ? operation.SpecificId : operation.NestedSpecificId);
                        operation.BackwardAdjacentOperations.Add(op);
                        parameters[j] = op;
                    }
                    else if (finder is IOperationBase[] opArray)
                    {
                        foreach (var op2 in opArray)
                        {
                            op2.Outputs.Add(layerInfo.Type == LayerInfoType.Normal ? operation.SpecificId : operation.NestedSpecificId);
                            operation.BackwardAdjacentOperations.Add(op2);
                        }

                        parameters[j] = opArray;
                    }
                    else
                    {
                        operation.BackwardAdjacentOperations.Add(null);
                        parameters[j] = finder;
                    }
                }
                else if (this.weightsAndBiases.ContainsKey(inputName))
                {
                    // Get the corresponding value from the dictionary using the input name
                    var p = this.weightsAndBiases[inputName](operation.LayerInfo);
                    operation.BackwardAdjacentOperations.Add(null);
                    parameters[j] = p;
                }
                else if (this.intermediates.ContainsKey(inputName))
                {
                    // Get the corresponding value from the dictionary using the input name
                    var p = this.intermediates[inputName](operation.LayerInfo);
                    operation.BackwardAdjacentOperations.Add(null);
                    parameters[j] = p;
                }
                else if (this.scalars.ContainsKey(inputName))
                {
                    // Get the corresponding value from the dictionary using the input name
                    var p = this.scalars[inputName](operation.LayerInfo);
                    operation.BackwardAdjacentOperations.Add(null);
                    parameters[j] = p;
                }
                else
                {
                    throw new InvalidOperationException($"Input name {inputName} not found in value map");
                }
            }

            operation.Parameters = parameters;
        }
    }
}

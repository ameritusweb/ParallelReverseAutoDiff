﻿//------------------------------------------------------------------------------
// <copyright file="ComputationGraph.cs" author="ameritusweb" date="5/8/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Concurrent;

    /// <summary>
    /// A computation graph.
    /// </summary>
    public abstract class ComputationGraph
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.RMAD";
        private ConcurrentDictionary<string, Func<LayerInfo, Matrix>> weights = new ConcurrentDictionary<string, Func<LayerInfo, Matrix>>();
        private ConcurrentDictionary<string, Func<LayerInfo, Matrix>> gradients = new ConcurrentDictionary<string, Func<LayerInfo, Matrix>>();
        private ConcurrentDictionary<string, Func<LayerInfo, Matrix>> intermediates = new ConcurrentDictionary<string, Func<LayerInfo, Matrix>>();
        private ConcurrentDictionary<string, Func<LayerInfo, IOperation>> operationFinders = new ConcurrentDictionary<string, Func<LayerInfo, IOperation>>();
        private ConcurrentDictionary<string, IOperation> operations = new ConcurrentDictionary<string, IOperation>();
        private IOperation? startOperation;
        private IOperation? currentOperation;
        private NeuralNetwork neuralNetwork;

        /// <summary>
        /// Initializes a new instance of the <see cref="ComputationGraph"/> class.
        /// </summary>
        /// <param name="neuralNetwork">The neural network.</param>
        public ComputationGraph(NeuralNetwork neuralNetwork)
        {
            this.neuralNetwork = neuralNetwork;
        }

        /// <summary>
        /// Gets the start operation.
        /// </summary>
        public IOperation? StartOperation
        {
            get
            {
                return this.startOperation;
            }
        }

        /// <summary>
        /// Gets the current operation.
        /// </summary>
        public IOperation? CurrentOperation
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
        public IOperation this[string operationIdentifier]
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
        public Matrix this[MatrixType type, string identifier, LayerInfo index]
        {
            get
            {
                return type switch
                {
                    MatrixType.Weight => this.weights[identifier](index),
                    MatrixType.Gradient => this.gradients[identifier](index),
                    MatrixType.Intermediate => this.intermediates[identifier](index),
                    _ => throw new ArgumentException("Invalid matrix type."),
                };
            }
        }

        /// <summary>
        /// Construct the computation graph from an architecture with no layers.
        /// </summary>
        /// <param name="architecture">The architecture.</param>
        /// <returns>The computation graph.</returns>
        public ComputationGraph ConstructFromArchitecture(JsonArchitecture architecture)
        {
            LayerInfo layerInfo = default(LayerInfo);
            foreach (var timeStep in architecture.TimeSteps)
            {
                foreach (var operationInfo in timeStep.StartOperations)
                {
                    this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                }

                foreach (var operationInfo in timeStep.EndOperations)
                {
                    this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
                }
            }

            return this;
        }

        /// <summary>
        /// Construct the computation graph from an architecture with layers.
        /// </summary>
        /// <param name="architecture">The architecture.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <returns>The computation graph.</returns>
        public ComputationGraph ConstructFromArchitecture(JsonArchitecture architecture, int numLayers)
        {
            var layerInfo = default(LayerInfo);
            foreach (var timeStep in architecture.TimeSteps)
            {
                foreach (var operationInfo in timeStep.StartOperations)
                {
                    this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
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

                foreach (var operationInfo in timeStep.EndOperations)
                {
                    this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
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
            var layerInfo = default(LayerInfo);
            for (int t = 0; t < numTimeSteps; t++)
            {
                layerInfo.TimeStep = t;
                foreach (var timeStep in architecture.TimeSteps)
                {
                    foreach (var operationInfo in timeStep.StartOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
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

                    foreach (var operationInfo in timeStep.EndOperations)
                    {
                        this.AddOperationByType(this.GetTypeFrom(operationInfo.Type), operationInfo, layerInfo);
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
        /// Adds a operation finder to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="operationFinder">The operation finder.</param>
        /// <returns>A computation graph.</returns>
        public ComputationGraph AddOperationFinder(string identifier, Func<LayerInfo, IOperation> operationFinder)
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

            IOperation operation = (IOperation)(instantiate.Invoke(null, new object[] { this.neuralNetwork }) ?? throw new Exception("Instantiate method should return a non-null operation."));
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
        public ComputationGraph AddOperation(IOperation operation, OperationInfo info, LayerInfo layerInfo)
        {
            this.OperationAdded(operation, info);
            this.InitializeOperation(operation, info, layerInfo);
            this.operations.TryAdd(operation.SpecificId, operation);
            this.SetupDependencies(operation);
            return this;
        }

        /// <summary>
        /// Initialize the computation graph operation.
        /// </summary>
        /// <param name="operation">The operation.</param>
        /// <param name="info">The operation info.</param>
        /// <param name="layerInfo">The layer info.</param>
        protected void InitializeOperation(IOperation operation, OperationInfo info, LayerInfo layerInfo)
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
            this.weights.TryAdd(identifier, matrix);
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
        /// Lifecycle function for when an intermediate is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="matrix">The gradient.</param>
        protected virtual void IntermediateAdded(string identifier, Func<LayerInfo, Matrix> matrix)
        {
            this.intermediates.TryAdd(identifier, matrix);
        }

        /// <summary>
        /// Lifecycle function for when an operation finder is added to the computation graph.
        /// </summary>
        /// <param name="identifier">An identifier.</param>
        /// <param name="operationFinder">The gradient.</param>
        protected virtual void OperationFinderAdded(string identifier, Func<LayerInfo, IOperation> operationFinder)
        {
            this.operationFinders.TryAdd(identifier, operationFinder);
        }

        /// <summary>
        /// Lifecycle function for when an operation is added to the computation graph.
        /// </summary>
        /// <param name="operation">The operation to add.</param>
        /// <param name="info">The operation info.</param>
        protected virtual void OperationAdded(IOperation operation, OperationInfo info)
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
        protected virtual void OperationInitialized(IOperation operation, OperationInfo info, LayerInfo layerInfo)
        {
            operation.InitializeFrom(info, this.gradients, layerInfo);
        }

        /// <summary>
        /// Setup dependencies for the operation.
        /// </summary>
        /// <param name="operation">The operation to setup.</param>
        protected void SetupDependencies(IOperation operation)
        {
            this.DependenciesSetup(operation);
        }

        /// <summary>
        /// Lifecycle method for when the dependencies are setup for an operation.
        /// </summary>
        /// <param name="operation">The operation.</param>
        protected virtual void DependenciesSetup(IOperation operation)
        {
            object[] parameters = new object[operation.Inputs.Count];
            for (int j = 0; j < operation.Inputs.Count; ++j)
            {
                var input = operation.Inputs[j];
                var split = input.Split(new[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                var inputName = split[0];
                var specificId = inputName + operation.LayerInfo.ToString();
                if (this.operations.ContainsKey(specificId))
                {
                    var inputOp = this.operations[specificId];
                    inputOp.Outputs.Add(operation.SpecificId);
                    operation.BackwardAdjacentOperations.Add(inputOp);
                    parameters[j] = inputOp;
                    continue;
                }

                if (this.weights.ContainsKey(inputName))
                {
                    // Get the corresponding value from the dictionary using the input name
                    var p = this.weights[inputName](operation.LayerInfo);
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
                else if (this.operationFinders.ContainsKey(inputName))
                {
                    // Get the corresponding value from the dictionary using the input name
                    var op = this.operationFinders[inputName](operation.LayerInfo);
                    op.Outputs.Add(op.SpecificId);
                    operation.BackwardAdjacentOperations.Add(op);
                    parameters[j] = op;
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
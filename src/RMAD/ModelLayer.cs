//------------------------------------------------------------------------------
// <copyright file="ModelLayer.cs" author="ameritusweb" date="5/27/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Concurrent;

    /// <summary>
    /// A model layer for a neural network.
    /// </summary>
    internal class ModelLayer : IModelLayer
    {
        private readonly ConcurrentDictionary<string, (object, object, object, object, int[], InitializationType)> elements = new ConcurrentDictionary<string, (object, object, object, object, int[], InitializationType)>();
        private readonly NeuralNetwork neuralNetwork;

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelLayer"/> class.
        /// </summary>
        /// <param name="neuralNetwork">The neural network.</param>
        internal ModelLayer(NeuralNetwork neuralNetwork)
        {
            this.neuralNetwork = neuralNetwork;
        }

        /// <summary>
        /// Gets the model elements dictionary.
        /// </summary>
        internal ConcurrentDictionary<string, (object weight, object gradient, object firstMoment, object secondMoment, int[] dimensions, InitializationType initialization)> Elements
        {
            get
            {
                return this.elements;
            }
        }

        /// <summary>
        /// Retrieve the model element by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <param name="elementType">The element type.</param>
        /// <returns>The model element.</returns>
        public object this[string identifier, ModelElementType elementType]
        {
            get
            {
                var tuple = this.elements[identifier];
                return elementType switch
                {
                    ModelElementType.Weight => tuple.Item1,
                    ModelElementType.Gradient => tuple.Item2,
                    ModelElementType.FirstMoment => tuple.Item3,
                    ModelElementType.SecondMoment => tuple.Item4,
                    _ => throw new ArgumentException("Invalid model element type."),
                };
            }
        }

        /// <summary>
        /// Retrieve the weight by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The weight.</returns>
        public Matrix? WeightMatrix(string identifier)
        {
            return this.elements[identifier].Item1 as Matrix;
        }

        /// <summary>
        /// Retrieve the weight by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The weight.</returns>
        public DeepMatrix? WeightDeepMatrix(string identifier)
        {
            return this.elements[identifier].Item1 as DeepMatrix;
        }

        /// <summary>
        /// Retrieve the weight by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The weight.</returns>
        public DeepMatrix[]? WeightDeepMatrixArray(string identifier)
        {
            return this.elements[identifier].Item1 as DeepMatrix[];
        }

        /// <summary>
        /// Retrieve the gradient by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The gradient.</returns>
        public Matrix? GradientMatrix(string identifier)
        {
            return this.elements[identifier].Item1 as Matrix;
        }

        /// <summary>
        /// Retrieve the gradient by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The gradient.</returns>
        public DeepMatrix? GradientDeepMatrix(string identifier)
        {
            return this.elements[identifier].Item1 as DeepMatrix;
        }

        /// <summary>
        /// Retrieve the gradient by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The gradient.</returns>
        public DeepMatrix[]? GradientDeepMatrixArray(string identifier)
        {
            return this.elements[identifier].Item1 as DeepMatrix[];
        }
    }
}

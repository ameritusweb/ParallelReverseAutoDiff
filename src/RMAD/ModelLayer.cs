//------------------------------------------------------------------------------
// <copyright file="ModelLayer.cs" author="ameritusweb" date="5/27/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;

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

        /// <inheritdoc />
        public List<string> Identifiers
        {
            get
            {
                return this.elements.Keys.ToList();
            }
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

        /// <inheritdoc />
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

        /// <inheritdoc />
        public Matrix? WeightMatrix(string identifier)
        {
            return this.elements[identifier].Item1 as Matrix;
        }

        /// <inheritdoc />
        public DeepMatrix? WeightDeepMatrix(string identifier)
        {
            return this.elements[identifier].Item1 as DeepMatrix;
        }

        /// <inheritdoc />
        public DeepMatrix[]? WeightDeepMatrixArray(string identifier)
        {
            return this.elements[identifier].Item1 as DeepMatrix[];
        }

        /// <inheritdoc />
        public Matrix? GradientMatrix(string identifier)
        {
            return this.elements[identifier].Item2 as Matrix;
        }

        /// <inheritdoc />
        public DeepMatrix? GradientDeepMatrix(string identifier)
        {
            return this.elements[identifier].Item2 as DeepMatrix;
        }

        /// <inheritdoc />
        public DeepMatrix[]? GradientDeepMatrixArray(string identifier)
        {
            return this.elements[identifier].Item2 as DeepMatrix[];
        }

        /// <inheritdoc />
        public int[]? Dimensions(string identifier)
        {
            return this.elements[identifier].Item5 as int[];
        }
    }
}

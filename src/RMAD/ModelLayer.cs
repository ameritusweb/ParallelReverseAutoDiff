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
    using System.IO;
    using System.Linq;
    using Newtonsoft.Json;

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
        public Matrix WeightMatrix(string identifier)
        {
            return this.elements[identifier].Item1 as Matrix ?? throw new ArgumentException("Identifier not found or wrong type.");
        }

        /// <inheritdoc />
        public DeepMatrix WeightDeepMatrix(string identifier)
        {
            return this.elements[identifier].Item1 as DeepMatrix ?? throw new ArgumentException("Identifier not found or wrong type.");
        }

        /// <inheritdoc />
        public DeepMatrix[] WeightDeepMatrixArray(string identifier)
        {
            return this.elements[identifier].Item1 as DeepMatrix[] ?? throw new ArgumentException("Identifier not found or wrong type.");
        }

        /// <inheritdoc />
        public Matrix GradientMatrix(string identifier)
        {
            return this.elements[identifier].Item2 as Matrix ?? throw new ArgumentException("Identifier not found or wrong type.");
        }

        /// <inheritdoc />
        public DeepMatrix GradientDeepMatrix(string identifier)
        {
            return this.elements[identifier].Item2 as DeepMatrix ?? throw new ArgumentException("Identifier not found or wrong type.");
        }

        /// <inheritdoc />
        public DeepMatrix[] GradientDeepMatrixArray(string identifier)
        {
            return this.elements[identifier].Item2 as DeepMatrix[] ?? throw new ArgumentException("Identifier not found or wrong type.");
        }

        /// <inheritdoc />
        public int[] Dimensions(string identifier)
        {
            return this.elements[identifier].Item5 as int[] ?? throw new ArgumentException("Identifier not found or wrong type.");
        }

        /// <inheritdoc />
        public IModelLayer Clone()
        {
            ModelLayer clone = new ModelLayer(this.neuralNetwork);
            foreach (var element in this.elements)
            {
                object weightClone = this.CloneHelper(element.Value.Item1);
                object gradientClone = this.CloneHelper(element.Value.Item2);
                object firstMomentClone = this.CloneHelper(element.Value.Item3);
                object secondMomentClone = this.CloneHelper(element.Value.Item4);
                clone.elements.TryAdd(element.Key, (weightClone, gradientClone, firstMomentClone, secondMomentClone, element.Value.Item5, element.Value.Item6));
            }

            return clone;
        }

        /// <inheritdoc />
        public void Average(IModelLayer layer)
        {
            IAverageableModelLayer other = (IAverageableModelLayer)layer;
            foreach (var element in this.elements)
            {
                string id = element.Key;
                if (other.Elements.TryGetValue(id, out var otherElement))
                {
                    this.elements[id] = (this.AverageHelper(element.Value.Item1, otherElement.Item1),
                                         this.AverageHelper(element.Value.Item2, otherElement.Item2),
                                         this.AverageHelper(element.Value.Item3, otherElement.Item3),
                                         this.AverageHelper(element.Value.Item4, otherElement.Item4),
                                         element.Value.Item5, element.Value.Item6);
                }
            }
        }

        /// <inheritdoc />
        public void AverageGradients(IModelLayer layer)
        {
            IAverageableModelLayer other = (IAverageableModelLayer)layer;
            foreach (var element in this.elements)
            {
                string id = element.Key;
                if (other.Elements.TryGetValue(id, out var otherElement))
                {
                    this.elements[id] = (element.Value.Item1,
                                         this.AverageHelper(element.Value.Item2, otherElement.Item2),
                                         element.Value.Item3,
                                         element.Value.Item4,
                                         element.Value.Item5, element.Value.Item6);
                }
            }
        }

        /// <inheritdoc />
        public void SaveWeights(FileInfo file)
        {
            // Create a new dictionary with only weights
            var weights = this.elements.ToDictionary(e => e.Key, e => e.Value.Item1);

            // Serialize the weights dictionary to JSON
            var json = JsonConvert.SerializeObject(weights, Formatting.Indented);

            // Write JSON string to file
            File.WriteAllText(file.FullName, json);
        }

        /// <inheritdoc />
        public void LoadWeights(FileInfo file)
        {
            // Read JSON string from file
            var json = File.ReadAllText(file.FullName);

            // Deserialize JSON string to a dictionary
            var weights = JsonConvert.DeserializeObject<Dictionary<string, object>>(json) ?? throw new InvalidOperationException("Weights must not be null.");

            // Update the weights in the elements dictionary
            foreach (var element in weights)
            {
                if (this.elements.TryGetValue(element.Key, out var value))
                {
                    this.elements[element.Key] = (element.Value, value.Item2, value.Item3, value.Item4, value.Item5, value.Item6);
                }
            }
        }

        /// <inheritdoc />
        public void ApplyGradients(List<object> gradients)
        {
            for (int i = 0; i < gradients.Count; ++i)
            {
                var key = this.elements.Keys.ElementAt(i);
                var tuple = this.elements[key];
                if (tuple.Item2 is Matrix matrix)
                {
                    if (gradients[i] is Matrix gradientMatrix)
                    {
                        matrix.Replace(gradientMatrix.ToArray());
                    }
                }
                else if (tuple.Item2 is DeepMatrix deepMatrix)
                {
                    if (gradients[i] is DeepMatrix gradientDeepMatrix)
                    {
                        deepMatrix.Replace(gradientDeepMatrix.ToArray());
                    }
                }
                else if (tuple.Item2 is DeepMatrix[] deepMatrixArray)
                {
                    if (gradients[i] is DeepMatrix[] gradientDeepMatrixArray)
                    {
                        for (int j = 0; j < deepMatrixArray.Length; ++j)
                        {
                            deepMatrixArray[j].Replace(gradientDeepMatrixArray[j].ToArray());
                        }
                    }
                }
                else
                {
                    throw new ArgumentException("Unsupported gradient type.");
                }
            }
        }

        /// <inheritdoc />
        public void ApplyWeights(List<object> weights)
        {
            for (int i = 0; i < weights.Count; ++i)
            {
                var key = this.elements.Keys.ElementAt(i);
                var tuple = this.elements[key];
                if (tuple.Item1 is Matrix matrix)
                {
                    if (weights[i] is Matrix weightMatrix)
                    {
                        matrix.Replace(weightMatrix.ToArray());
                    }
                }
                else if (tuple.Item1 is DeepMatrix deepMatrix)
                {
                    if (weights[i] is DeepMatrix weightDeepMatrix)
                    {
                        deepMatrix.Replace(weightDeepMatrix.ToArray());
                    }
                }
                else if (tuple.Item1 is DeepMatrix[] deepMatrixArray)
                {
                    if (weights[i] is DeepMatrix[] weightDeepMatrixArray)
                    {
                        for (int j = 0; j < deepMatrixArray.Length; ++j)
                        {
                            deepMatrixArray[j].Replace(weightDeepMatrixArray[j].ToArray());
                        }
                    }
                }
                else
                {
                    throw new ArgumentException("Unsupported weight type.");
                }
            }
        }

        private object CloneHelper(object element)
        {
            switch (element)
            {
                case Matrix m:
                    return m.Clone();
                case DeepMatrix dm:
                    return dm.Clone();
                case DeepMatrix[] dma:
                    return dma.Select(x => x.Clone()).ToArray();
                default:
                    throw new ArgumentException("Unsupported element type.");
            }
        }

        private object AverageHelper(object element1, object element2)
        {
            switch (element1, element2)
            {
                case (Matrix m1, Matrix m2):
                    return m1.Average(m2);
                case (DeepMatrix dm1, DeepMatrix dm2):
                    return dm1.Average(dm2);
                case (DeepMatrix[] dma1, DeepMatrix[] dma2):
                    if (dma1.Length != dma2.Length)
                    {
                        throw new ArgumentException("Incompatible DeepMatrix array lengths.");
                    }

                    return dma1.Zip(dma2, (dm1, dm2) => dm1.Average(dm2)).ToArray();
                default:
                    throw new ArgumentException("Unsupported element type or incompatible elements.");
            }
        }
    }
}

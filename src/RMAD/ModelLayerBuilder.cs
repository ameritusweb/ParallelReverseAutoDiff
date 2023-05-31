//------------------------------------------------------------------------------
// <copyright file="ModelLayerBuilder.cs" author="ameritusweb" date="5/27/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// A model layer builder for a neural network.
    /// </summary>
    public class ModelLayerBuilder
    {
        private readonly ModelLayer modelLayer;

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelLayerBuilder"/> class.
        /// </summary>
        /// <param name="neuralNetwork">The neural network.</param>
        public ModelLayerBuilder(NeuralNetwork neuralNetwork)
        {
            this.modelLayer = new ModelLayer(neuralNetwork);
        }

        /// <summary>
        /// Add model element group.
        /// </summary>
        /// <param name="id">The identifier.</param>
        /// <param name="dimensions">The dimensions.</param>
        /// <param name="initialization">The type of initialization.</param>
        /// <returns>The model layer builder.</returns>
        public ModelLayerBuilder AddModelElementGroup(string id, int[] dimensions, InitializationType initialization)
        {
            switch (dimensions.Length)
            {
                case 2:
                    {
                        var weight = new Matrix(dimensions[0], dimensions[1]);
                        weight.Initialize(initialization);
                        var gradient = new Matrix(dimensions[0], dimensions[1]);
                        var firstMoment = new Matrix(dimensions[0], dimensions[1]);
                        var secondMoment = new Matrix(dimensions[0], dimensions[1]);
                        this.modelLayer.Elements.TryAdd(id, (weight, gradient, firstMoment, secondMoment, dimensions, initialization));
                        break;
                    }

                case 3:
                    {
                        var weight = new DeepMatrix(dimensions[0], dimensions[1], dimensions[2]);
                        weight.Initialize(initialization);
                        var gradient = new DeepMatrix(dimensions[0], dimensions[1], dimensions[2]);
                        var firstMoment = new DeepMatrix(dimensions[0], dimensions[1], dimensions[2]);
                        var secondMoment = new DeepMatrix(dimensions[0], dimensions[1], dimensions[2]);
                        this.modelLayer.Elements.TryAdd(id, (weight, gradient, firstMoment, secondMoment, dimensions, initialization));
                        break;
                    }

                case 4:
                    {
                        DeepMatrix[] weight = new DeepMatrix[dimensions[0]];
                        for (int f = 0; f < dimensions[0]; ++f)
                        {
                            weight[f] = new DeepMatrix(dimensions[1], dimensions[2], dimensions[3]);
                            weight[f].Initialize(initialization);
                        }

                        DeepMatrix[] gradient = new DeepMatrix[dimensions[0]];
                        for (int f = 0; f < dimensions[0]; ++f)
                        {
                            gradient[f] = new DeepMatrix(dimensions[1], dimensions[2], dimensions[3]);
                        }

                        DeepMatrix[] firstMoment = new DeepMatrix[dimensions[0]];
                        for (int f = 0; f < dimensions[0]; ++f)
                        {
                            firstMoment[f] = new DeepMatrix(dimensions[1], dimensions[2], dimensions[3]);
                        }

                        DeepMatrix[] secondMoment = new DeepMatrix[dimensions[0]];
                        for (int f = 0; f < dimensions[0]; ++f)
                        {
                            secondMoment[f] = new DeepMatrix(dimensions[1], dimensions[2], dimensions[3]);
                        }

                        this.modelLayer.Elements.TryAdd(id, (weight, gradient, firstMoment, secondMoment, dimensions, initialization));
                        break;
                    }

                default:
                    throw new ArgumentException("Invalid dimensions.");
            }

            return this;
        }

        /// <summary>
        /// Builds the model layer.
        /// </summary>
        /// <returns>The model layer.</returns>
        public IModelLayer Build()
        {
            return this.modelLayer;
        }
    }
}

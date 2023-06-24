//------------------------------------------------------------------------------
// <copyright file="GradientStore.cs" author="ameritusweb" date="6/23/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;

    /// <summary>
    /// A gradient store.
    /// </summary>
    [Serializable]
    public class GradientStore : StoreBase
    {
        /// <summary>
        /// Average the gradients from the gradient stores.
        /// </summary>
        /// <param name="gradientStores">The gradient stores.</param>
        /// <returns>The averaged gradients.</returns>
        public static GradientStore AverageGradients(IEnumerable<GradientStore> gradientStores)
        {
            // Instantiate a new gradient store for the averaged gradients
            GradientStore averageGradientStore = new GradientStore();

            // Create a list from the enumerable to avoid multiple enumerations
            List<GradientStore> gradientStoresList = gradientStores.ToList();

            // If the list is not empty, copy the Ids, Types, and ModelLayerIndices from the first gradient store
            if (gradientStoresList.Count > 0)
            {
                var firstGradientStore = gradientStoresList[0];
                averageGradientStore.Ids = new List<Guid>(firstGradientStore.Ids);
                averageGradientStore.Types = new List<string>(firstGradientStore.Types);
                averageGradientStore.ModelLayerIndices = new List<(int, int)>(firstGradientStore.ModelLayerIndices);
            }

            // For each gradient store in the collection
            foreach (var gradientStore in gradientStoresList)
            {
                // Iterate through each id in the gradient store
                for (int i = 0; i < gradientStore.Ids.Count; i++)
                {
                    var id = gradientStore.Ids[i];
                    var type = gradientStore.Types[i];

                    switch (type)
                    {
                        case "matrix":
                            if (averageGradientStore.Matrices.ContainsKey(id))
                            {
                                averageGradientStore.Matrices[id] =
                                    averageGradientStore.Matrices[id].Average(gradientStore.Matrices[id]);
                            }
                            else
                            {
                                averageGradientStore.Matrices.TryAdd(id, gradientStore.Matrices[id]);
                            }

                            break;

                        case "deepMatrix":
                            if (averageGradientStore.DeepMatrices.ContainsKey(id))
                            {
                                averageGradientStore.DeepMatrices[id] =
                                    averageGradientStore.DeepMatrices[id].Average(gradientStore.DeepMatrices[id]);
                            }
                            else
                            {
                                averageGradientStore.DeepMatrices.TryAdd(id, gradientStore.DeepMatrices[id]);
                            }

                            break;

                        case "deepMatrixArray":
                            if (averageGradientStore.DeepMatrixArrays.ContainsKey(id))
                            {
                                var averagedArray = averageGradientStore.DeepMatrixArrays[id];
                                var newArray = gradientStore.DeepMatrixArrays[id];
                                for (int j = 0; j < averagedArray.Length; ++j)
                                {
                                    averagedArray[j] = averagedArray[j].Average(newArray[j]);
                                }

                                averageGradientStore.DeepMatrixArrays[id] = averagedArray;
                            }
                            else
                            {
                                averageGradientStore.DeepMatrixArrays.TryAdd(id, gradientStore.DeepMatrixArrays[id]);
                            }

                            break;

                        default:
                            throw new InvalidOperationException($"Gradient type '{type}' not found.");
                    }
                }
            }

            return averageGradientStore;
        }

        /// <summary>
        /// Loads a gradient store from a file.
        /// </summary>
        /// <param name="fileInfo">The file info.</param>
        /// <returns>The gradient store.</returns>
        public static GradientStore LoadNew(FileInfo fileInfo)
        {
            GradientStore store = new GradientStore();
            store.InternalLoad(fileInfo);
            return store;
        }

        /// <summary>
        /// Loads a gradient store from a file.
        /// </summary>
        /// <param name="fileInfo">The file info.</param>
        public void Load(FileInfo fileInfo)
        {
            this.InternalClear();
            this.InternalLoad(fileInfo);
        }

        /// <summary>
        /// Retrieves the gradients for a model layer.
        /// </summary>
        /// <param name="index">The index of the model layer.</param>
        /// <returns>The model layer gradients.</returns>
        public List<object> ToModelLayerGradients(int index)
        {
            var indices = this.ModelLayerIndices[index];
            var ids = this.Ids.GetRange(indices.Item1, indices.Item2 - indices.Item1 + 1);
            var types = this.Types.GetRange(indices.Item1, indices.Item2 - indices.Item1 + 1);
            List<object> gradients = new List<object>();
            for (int i = 0; i < ids.Count; ++i)
            {
                var id = ids[i];
                var type = types[i];
                switch (type)
                {
                    case "matrix":
                        gradients.Add(this.Matrices[id]);
                        break;
                    case "deepMatrix":
                        gradients.Add(this.DeepMatrices[id]);
                        break;
                    case "deepMatrixArray":
                        gradients.Add(this.DeepMatrixArrays[id]);
                        break;
                    default:
                        throw new InvalidOperationException("Type not found.");
                }
            }

            return gradients;
        }

        /// <summary>
        /// Adds a model layer.
        /// </summary>
        /// <param name="modelLayer">The model layer.</param>
        public override void Add(IModelLayer modelLayer)
        {
            var index = this.Ids.Count;
            foreach (var identifier in modelLayer.Identifiers)
            {
                var gradient = modelLayer[identifier, ModelElementType.Gradient];
                if (gradient is Matrix matrix)
                {
                    this.Add(matrix);
                }
                else if (gradient is DeepMatrix deepMatrix)
                {
                    this.Add(deepMatrix);
                }
                else if (gradient is DeepMatrix[] deepMatrixArray)
                {
                    this.Add(deepMatrixArray);
                }
            }

            var lastIndex = this.Ids.Count - 1;
            this.ModelLayerIndices.Add((index, lastIndex));
        }
    }
}

//------------------------------------------------------------------------------
// <copyright file="WeightStore.cs" author="ameritusweb" date="6/23/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using Newtonsoft.Json;

    /// <summary>
    /// A weight store.
    /// </summary>
    [Serializable]
    public class WeightStore : StoreBase
    {
        /// <summary>
        /// Loads a weight store from a file.
        /// </summary>
        /// <param name="fileInfo">The file info.</param>
        /// <returns>The weight store.</returns>
        public static WeightStore LoadNew(FileInfo fileInfo)
        {
            WeightStore store = new WeightStore();
            store.InternalLoad(fileInfo);
            return store;
        }

        /// <summary>
        /// Loads a weight store from a file.
        /// </summary>
        /// <param name="fileInfo">The file info.</param>
        public void Load(FileInfo fileInfo)
        {
            this.InternalClear();
            this.InternalLoad(fileInfo);
        }

        /// <summary>
        /// Retrieves the weights for a model layer.
        /// </summary>
        /// <param name="index">The index of the model layer.</param>
        /// <returns>The model layer weights.</returns>
        public List<object> ToModelLayerWeights(int index)
        {
            var indices = this.ModelLayerIndices[index];
            var ids = this.Ids.GetRange(indices.Item1, indices.Item2 - indices.Item1 + 1);
            var types = this.Types.GetRange(indices.Item1, indices.Item2 - indices.Item1 + 1);
            List<object> weights = new List<object>();
            for (int i = 0; i < ids.Count; ++i)
            {
                var id = ids[i];
                var type = types[i];
                switch (type)
                {
                    case "matrix":
                        weights.Add(this.Matrices[id]);
                        break;
                    case "deepMatrix":
                        weights.Add(this.DeepMatrices[id]);
                        break;
                    case "deepMatrixArray":
                        weights.Add(this.DeepMatrixArrays[id]);
                        break;
                    default:
                        throw new InvalidOperationException("Type not found.");
                }
            }

            return weights;
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
                var weight = modelLayer[identifier, ModelElementType.Weight];
                if (weight is Matrix matrix)
                {
                    this.Add(matrix);
                }
                else if (weight is DeepMatrix deepMatrix)
                {
                    this.Add(deepMatrix);
                }
                else if (weight is DeepMatrix[] deepMatrixArray)
                {
                    this.Add(deepMatrixArray);
                }
            }

            var lastIndex = this.Ids.Count - 1;
            this.ModelLayerIndices.Add((index, lastIndex));
        }
    }
}

//------------------------------------------------------------------------------
// <copyright file="GradientStore.cs" author="ameritusweb" date="6/23/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.IO;
    using Newtonsoft.Json;

    /// <summary>
    /// A gradient store.
    /// </summary>
    [Serializable]
    public class GradientStore
    {
        /// <summary>
        /// Gets or sets the matrix gradients.
        /// </summary>
        public ConcurrentDictionary<Guid, Matrix> MatrixGradients { get; set; } = new ConcurrentDictionary<Guid, Matrix>();

        /// <summary>
        /// Gets or sets the deep matrix gradients.
        /// </summary>
        public ConcurrentDictionary<Guid, DeepMatrix> DeepMatrixGradients { get; set; } = new ConcurrentDictionary<Guid, DeepMatrix>();

        /// <summary>
        /// Gets or sets the deep matrix array gradients.
        /// </summary>
        public ConcurrentDictionary<Guid, DeepMatrix[]> DeepMatrixArrayGradients { get; set; } = new ConcurrentDictionary<Guid, DeepMatrix[]>();

        /// <summary>
        /// Gets or sets the ids.
        /// </summary>
        public List<Guid> Ids { get; set; } = new List<Guid>();

        /// <summary>
        /// Gets or sets the types.
        /// </summary>
        public List<string> Types { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets the model layer indices.
        /// </summary>
        public List<(int, int)> ModelLayerIndices { get; set; } = new List<(int, int)>();

        /// <summary>
        /// Loads a gradient store from a file.
        /// </summary>
        /// <param name="fileInfo">The file info.</param>
        /// <returns>The gradient store.</returns>
        public static GradientStore Load(FileInfo fileInfo)
        {
            // Read JSON string from file
            var json = File.ReadAllText(fileInfo.FullName);

            // Deserialize JSON string to a gradient store
            var store = JsonConvert.DeserializeObject<GradientStore>(json) ?? throw new InvalidOperationException("An error occurred during deserialization.");

            return store;
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
                        gradients.Add(this.MatrixGradients[id]);
                        break;
                    case "deepMatrix":
                        gradients.Add(this.DeepMatrixGradients[id]);
                        break;
                    case "deepMatrixArray":
                        gradients.Add(this.DeepMatrixArrayGradients[id]);
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
        public void Add(IModelLayer modelLayer)
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

        /// <summary>
        /// Save gradients to a file.
        /// </summary>
        /// <param name="file">The file info.</param>
        public void Save(FileInfo file)
        {
            // Serialize the weights dictionary to JSON
            var json = JsonConvert.SerializeObject(this, Formatting.Indented);

            // Write JSON string to file
            File.WriteAllText(file.FullName, json);
        }

        /// <summary>
        /// Adds a matrix.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        private void Add(Matrix matrix)
        {
            Guid id = Guid.NewGuid();
            this.Ids.Add(id);
            this.Types.Add(nameof(matrix));
            this.MatrixGradients.AddOrUpdate(id, matrix, (key, oldValue) => matrix);
        }

        /// <summary>
        /// Adds a deep matrix.
        /// </summary>
        /// <param name="deepMatrix">The deep matrix.</param>
        private void Add(DeepMatrix deepMatrix)
        {
            Guid id = Guid.NewGuid();
            this.Ids.Add(id);
            this.Types.Add(nameof(deepMatrix));
            this.DeepMatrixGradients.AddOrUpdate(id, deepMatrix, (key, oldValue) => deepMatrix);
        }

        /// <summary>
        /// Adds a deep matrix array.
        /// </summary>
        /// <param name="deepMatrixArray">The deep matrix array.</param>
        private void Add(DeepMatrix[] deepMatrixArray)
        {
            Guid id = Guid.NewGuid();
            this.Ids.Add(id);
            this.Types.Add(nameof(deepMatrixArray));
            this.DeepMatrixArrayGradients.AddOrUpdate(id, deepMatrixArray, (key, oldValue) => deepMatrixArray);
        }
    }
}

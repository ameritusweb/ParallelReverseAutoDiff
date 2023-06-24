//------------------------------------------------------------------------------
// <copyright file="StoreBase.cs" author="ameritusweb" date="6/23/2023">
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
    /// A base store for the neural network.
    /// </summary>
    public abstract class StoreBase
    {
        /// <summary>
        /// Gets or sets the matrix gradients.
        /// </summary>
        public ConcurrentDictionary<Guid, Matrix> Matrices { get; set; } = new ConcurrentDictionary<Guid, Matrix>();

        /// <summary>
        /// Gets or sets the deep matrix gradients.
        /// </summary>
        public ConcurrentDictionary<Guid, DeepMatrix> DeepMatrices { get; set; } = new ConcurrentDictionary<Guid, DeepMatrix>();

        /// <summary>
        /// Gets or sets the deep matrix array gradients.
        /// </summary>
        public ConcurrentDictionary<Guid, DeepMatrix[]> DeepMatrixArrays { get; set; } = new ConcurrentDictionary<Guid, DeepMatrix[]>();

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
        /// Adds a model layer.
        /// </summary>
        /// <param name="modelLayer">The model layer to add.</param>
        public abstract void Add(IModelLayer modelLayer);

        /// <summary>
        /// Adds a range of model layers.
        /// </summary>
        /// <param name="modelLayers">The model layers.</param>
        public void AddRange(IEnumerable<IModelLayer> modelLayers)
        {
            foreach (var modelLayer in modelLayers)
            {
                this.Add(modelLayer);
            }
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
        protected void Add(Matrix matrix)
        {
            Guid id = Guid.NewGuid();
            this.Ids.Add(id);
            this.Types.Add(nameof(matrix));
            this.Matrices.AddOrUpdate(id, matrix, (key, oldValue) => matrix);
        }

        /// <summary>
        /// Adds a deep matrix.
        /// </summary>
        /// <param name="deepMatrix">The deep matrix.</param>
        protected void Add(DeepMatrix deepMatrix)
        {
            Guid id = Guid.NewGuid();
            this.Ids.Add(id);
            this.Types.Add(nameof(deepMatrix));
            this.DeepMatrices.AddOrUpdate(id, deepMatrix, (key, oldValue) => deepMatrix);
        }

        /// <summary>
        /// Adds a deep matrix array.
        /// </summary>
        /// <param name="deepMatrixArray">The deep matrix array.</param>
        protected void Add(DeepMatrix[] deepMatrixArray)
        {
            Guid id = Guid.NewGuid();
            this.Ids.Add(id);
            this.Types.Add(nameof(deepMatrixArray));
            this.DeepMatrixArrays.AddOrUpdate(id, deepMatrixArray, (key, oldValue) => deepMatrixArray);
        }
    }
}

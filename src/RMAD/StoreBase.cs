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
        /// Combines this store with another store.
        /// </summary>
        /// <param name="incomingStore">The incoming store.</param>
        /// <returns>The result store.</returns>
        public StoreBase Combine(StoreBase incomingStore)
        {
            if (incomingStore == null)
            {
                throw new ArgumentNullException(nameof(incomingStore));
            }

            // Ensure the two stores are of the same subtype
            if (this.GetType() != incomingStore.GetType())
            {
                throw new ArgumentException("The two stores must be of the same type.", nameof(incomingStore));
            }

            // Get the offset for ModelLayerIndices
            int offset = this.ModelLayerIndices[^1].Item2 + 1;

            // Copy all data from incoming store
            for (int i = 0; i < incomingStore.Ids.Count; i++)
            {
                var id = incomingStore.Ids[i];
                this.Ids.Add(id);
                this.Types.Add(incomingStore.Types[i]);

                // Check which dictionary the ID belongs to and add the corresponding item to this store
                if (incomingStore.Matrices.ContainsKey(id))
                {
                    this.Matrices.TryAdd(id, incomingStore.Matrices[id]);
                }
                else if (incomingStore.DeepMatrices.ContainsKey(id))
                {
                    this.DeepMatrices.TryAdd(id, incomingStore.DeepMatrices[id]);
                }
                else if (incomingStore.DeepMatrixArrays.ContainsKey(id))
                {
                    this.DeepMatrixArrays.TryAdd(id, incomingStore.DeepMatrixArrays[id]);
                }
            }

            // Adjust and copy the ModelLayerIndices from the incoming store
            foreach (var indexPair in incomingStore.ModelLayerIndices)
            {
                this.ModelLayerIndices.Add((indexPair.Item1 + offset, indexPair.Item2 + offset));
            }

            return this;
        }

        /// <summary>
        /// Save to a file.
        /// </summary>
        /// <param name="fileInfo">The file info.</param>
        public void Save(FileInfo fileInfo)
        {
            using (StreamWriter file = File.CreateText(fileInfo.FullName))
            using (JsonTextWriter writer = new JsonTextWriter(file))
            {
                JsonSerializer serializer = new JsonSerializer();

                // Start a JSON array
                writer.WriteStartArray();

                writer.WriteValue(Constants.IdsStart);
                writer.WriteStartArray();
                foreach (Guid id in this.Ids)
                {
                    serializer.Serialize(writer, id);
                }

                writer.WriteEndArray();

                writer.WriteValue(Constants.TypesStart);
                writer.WriteStartArray();
                foreach (string type in this.Types)
                {
                    serializer.Serialize(writer, type);
                }

                writer.WriteEndArray();

                writer.WriteValue(Constants.ModelLayerIndicesStart);
                writer.WriteStartArray();
                foreach (var indexPair in this.ModelLayerIndices)
                {
                    serializer.Serialize(writer, indexPair);
                }

                writer.WriteEndArray();

                writer.WriteValue(Constants.MatricesStart);
                writer.WriteStartArray();
                foreach (var matrixKey in this.Matrices.Keys)
                {
                    serializer.Serialize(writer, matrixKey);
                    serializer.Serialize(writer, this.Matrices[matrixKey]);
                }

                writer.WriteEndArray();

                writer.WriteValue(Constants.DeepMatricesStart);
                writer.WriteStartArray();
                foreach (var deepMatrixKey in this.DeepMatrices.Keys)
                {
                    serializer.Serialize(writer, deepMatrixKey);
                    serializer.Serialize(writer, this.DeepMatrices[deepMatrixKey]);
                }

                writer.WriteEndArray();

                writer.WriteValue(Constants.DeepMatrixArraysStart);
                writer.WriteStartArray();
                foreach (var deepMatrixArrayKey in this.DeepMatrixArrays.Keys)
                {
                    serializer.Serialize(writer, deepMatrixArrayKey);
                    serializer.Serialize(writer, this.DeepMatrixArrays[deepMatrixArrayKey]);
                }

                writer.WriteEndArray();

                // End the JSON array
                writer.WriteEndArray();
            }
        }

        /// <summary>
        /// Clears the store.
        /// </summary>
        protected void InternalClear()
        {
            this.Ids.Clear();
            this.Types.Clear();
            this.ModelLayerIndices.Clear();
            this.Matrices.Clear();
            this.DeepMatrices.Clear();
            this.DeepMatrixArrays.Clear();
        }

        /// <summary>
        /// Load from a file.
        /// </summary>
        /// <param name="fileInfo">The file info.</param>
        protected void InternalLoad(FileInfo fileInfo)
        {
            using (StreamReader file = File.OpenText(fileInfo.FullName))
            using (JsonTextReader reader = new JsonTextReader(file))
            {
                JsonSerializer serializer = new JsonSerializer();

                // The reader should be positioned at the start of an array
                if (!reader.Read() || reader.TokenType != JsonToken.StartArray)
                {
                    throw new InvalidDataException("Expected start of array");
                }

                // Read until we reach the end of the array
                while (reader.Read() && reader.TokenType != JsonToken.EndArray)
                {
                    // Expect a string to start each section
                    if (reader.TokenType != JsonToken.String)
                    {
                        throw new InvalidDataException("Expected section start marker");
                    }

                    string sectionStart = (string)reader.Value!;
                    reader.Read();

                    // Read the elements of the section
                    while (reader.Read() && reader.TokenType != JsonToken.EndArray)
                    {
                        switch (sectionStart)
                        {
                            case Constants.IdsStart:
                                if (reader.Value != null)
                                {
                                    this.Ids.Add(new Guid(reader.Value.ToString()));
                                }

                                break;

                            case Constants.TypesStart:
                                if (reader.Value != null)
                                {
                                    this.Types.Add(reader.Value.ToString());
                                }

                                break;

                            case Constants.ModelLayerIndicesStart:
                                var indices = serializer.Deserialize<(int, int)>(reader);
                                this.ModelLayerIndices.Add(indices);
                                break;

                            case Constants.MatricesStart:
                                Guid matrixKey = new Guid(reader.Value!.ToString());
                                reader.Read(); // Read the corresponding value
                                Matrix matrixValue = serializer.Deserialize<Matrix>(reader) ?? throw new InvalidOperationException("Matrix should not be null.");
                                this.Matrices.TryAdd(matrixKey, matrixValue);
                                break;

                            case Constants.DeepMatricesStart:
                                Guid deepMatrixKey = new Guid(reader.Value!.ToString());
                                reader.Read(); // Read the corresponding value
                                DeepMatrix deepMatrixValue = serializer.Deserialize<DeepMatrix>(reader) ?? throw new InvalidOperationException("DeepMatrix should not be null.");
                                this.DeepMatrices.TryAdd(deepMatrixKey, deepMatrixValue);
                                break;

                            case Constants.DeepMatrixArraysStart:
                                Guid deepMatrixArrayKey = new Guid(reader.Value!.ToString());
                                reader.Read(); // Read the corresponding value
                                DeepMatrix[] deepMatrixArrayValue = serializer.Deserialize<DeepMatrix[]>(reader) ?? throw new InvalidOperationException("DeepMatrix array should not be null.");
                                this.DeepMatrixArrays.TryAdd(deepMatrixArrayKey, deepMatrixArrayValue);
                                break;

                            default:
                                throw new InvalidDataException("Unknown section start marker: " + sectionStart);
                        }
                    }
                }
            }
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
            this.Matrices.AddOrUpdate(id, matrix, (_, _) => matrix);
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
            this.DeepMatrices.AddOrUpdate(id, deepMatrix, (_, _) => deepMatrix);
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
            this.DeepMatrixArrays.AddOrUpdate(id, deepMatrixArray, (_, _) => deepMatrixArray);
        }
    }
}

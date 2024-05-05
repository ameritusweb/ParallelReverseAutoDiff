// ------------------------------------------------------------------------------
// <copyright file="Maze.cs" author="ameritusweb" date="4/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample.VGruNetwork
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A maze.
    /// </summary>
    public class Maze
    {
        private IModelLayer inputLayer;
        private IModelLayer[] hiddenLayers;
        private IModelLayer outputLayer;

        /// <summary>
        /// Initializes a new instance of the <see cref="Maze"/> class.
        /// </summary>
        public Maze()
        {
        }

        /// <summary>
        /// Gets or sets the structure of the maze.
        /// </summary>
        public int[,] Structure { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] Weights { get; set; }

        /// <summary>
        /// Gets or sets the angles.
        /// </summary>
        public Matrix[] Angles { get; set; }

        /// <summary>
        /// Gets or sets the vectors.
        /// </summary>
        public Matrix[] Vectors { get; set; }

        /// <summary>
        /// Gets or sets the layers.
        /// </summary>
        public MazeLayer[] Layers { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] RowSumWeights { get; set; }

        /// <summary>
        /// Gets or sets the number of rows.
        /// </summary>
        public int NumRows { get; set; }

        /// <summary>
        /// Gets or sets the number of columns.
        /// </summary>
        public int NumCols { get; set; }

        /// <summary>
        /// Gets or sets the indices.
        /// </summary>
        public int[] Indices { get; set; }

        /// <summary>
        /// An indexer.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The matrix array.</returns>
        public Matrix[] this[string identifier]
        {
            get
            {
                switch (identifier)
                {
                    case "Weights":
                        return this.Weights;
                    case "Angles":
                        return this.Angles;
                    case "Vectors":
                        return this.Vectors;
                    case "RowSumWeights":
                        return this.RowSumWeights;
                    default:
                        throw new KeyNotFoundException();
                }
            }
        }

        /// <summary>
        /// Set the layers.
        /// </summary>
        /// <param name="inputLayer">The input layer.</param>
        /// <param name="hiddenLayers">The hidden layers.</param>
        /// <param name="outputLayer">The output layer.</param>
        public void SetLayers(IModelLayer inputLayer, IModelLayer[] hiddenLayers, IModelLayer outputLayer)
        {
            this.inputLayer = inputLayer;
            this.hiddenLayers = hiddenLayers;
            this.outputLayer = outputLayer;
        }

        /// <summary>
        /// Clone the maze.
        /// </summary>
        /// <returns>The cloned object.</returns>
        public object Clone()
        {
            var maze = new Maze
            {
                Structure = this.Structure,
                Weights = this.Weights,
                Angles = this.Angles,
                Vectors = this.Vectors,
                Layers = this.Layers,
                RowSumWeights = this.RowSumWeights,
                NumRows = this.NumRows,
                NumCols = this.NumCols,
                Indices = this.Indices,
            };
            maze.SetLayers(this.inputLayer, this.hiddenLayers, this.outputLayer);
            maze.SetIndices();
            return maze;
        }

        /// <summary>
        /// Sets the weights from the structure.
        /// </summary>
        public void SetWeightsFromStructure()
        {
            this.SetIndices();
            this.Layers = new MazeLayer[2];
            this.Layers[0] = new MazeLayer();
            this.Layers[1] = new MazeLayer();

            this.Weights = this.InitializeWeights(this.inputLayer, "Weights");
            this.Angles = this.InitializeWeights(this.inputLayer, "Angles");
            this.Vectors = this.InitializeWeights(this.inputLayer, "Vectors");

            for (int i = 0; i < this.hiddenLayers.Length; ++i)
            {
                this.Layers[i].UCWeights = this.InitializeWeights(this.hiddenLayers[i], "UCWeights");
                this.Layers[i].UpdateWeights = this.InitializeWeights(this.hiddenLayers[i], "UpdateWeights");
                this.Layers[i].ResetWeights = this.InitializeWeights(this.hiddenLayers[i], "ResetWeights");
                this.Layers[i].CandidateWeights = this.InitializeWeights(this.hiddenLayers[i], "CandidateWeights");
                this.Layers[i].HiddenWeights = this.InitializeWeights(this.hiddenLayers[i], "HiddenWeights");
                this.Layers[i].WUpdateWeights = this.InitializeWeights(this.hiddenLayers[i], "WUpdateWeights");
                this.Layers[i].UUpdateWeights = this.InitializeWeights(this.hiddenLayers[i], "UUpdateWeights");
                this.Layers[i].WUpdateVectors = this.InitializeWeightsVectors(this.hiddenLayers[i], "WUpdateVectors");
                this.Layers[i].UUpdateVectors = this.InitializeWeightsVectors(this.hiddenLayers[i], "UUpdateVectors");
                this.Layers[i].ZKeys = this.InitializeWeightsSquare(this.hiddenLayers[i], "ZKeys");
                this.Layers[i].ZKB = this.InitializeWeightsBias(this.hiddenLayers[i], "ZKB");
                this.Layers[i].CHKB = this.InitializeWeightsBias(this.hiddenLayers[i], "CHKB");
                this.Layers[i].WResetVectors = this.InitializeWeightsVectors(this.hiddenLayers[i], "WResetVectors");
                this.Layers[i].UResetVectors = this.InitializeWeightsVectors(this.hiddenLayers[i], "UResetVectors");
                this.Layers[i].WResetWeights = this.InitializeWeights(this.hiddenLayers[i], "WResetWeights");
                this.Layers[i].UResetWeights = this.InitializeWeights(this.hiddenLayers[i], "UResetWeights");
                this.Layers[i].CHKeys = this.InitializeWeightsSquare(this.hiddenLayers[i], "CHKeys");
                this.Layers[i].IKB = this.InitializeWeightsBias(this.hiddenLayers[i], "IKB");
                this.Layers[i].IKeys = this.InitializeWeightsSquare(this.hiddenLayers[i], "IKeys");
                this.Layers[i].RKB = this.InitializeWeightsBias(this.hiddenLayers[i], "RKB");
                this.Layers[i].RKeys = this.InitializeWeightsSquare(this.hiddenLayers[i], "RKeys");
            }

            this.RowSumWeights = this.InitializeWeights(this.outputLayer, "RowSumWeights");
        }

        /// <summary>
        /// Updates the model layers.
        /// </summary>
        public void UpdateModelLayers()
        {
            this.UpdateLayerGradients(this.inputLayer, -1, new string[] { "Weights", "Angles", "Vectors" });
            int i = 0;
            foreach (var layer in this.hiddenLayers)
            {
                this.UpdateLayerGradients(layer, i, new string[]
                {
                    "UCWeights", "UpdateWeights", "ResetWeights", "CandidateWeights",
                    "HiddenWeights", "WUpdateWeights", "UUpdateWeights",
                    "WResetWeights", "UResetWeights",
                });
                this.UpdateLayerGradientsVectors(layer, i, new string[]
                {
                    "WUpdateVectors",
                    "UUpdateVectors", "WResetVectors", "UResetVectors",
                });
                this.UpdateLayerGradientsBias(layer, i, new string[]
                {
                    "ZKB", "CHKB", "IKB", "RKB",
                });
                this.UpdateLayerGradientsSquare(layer, i, new string[]
                {
                    "ZKeys", "CHKeys", "IKeys", "RKeys",
                });
                i++;
            }

            this.UpdateLayerGradients(this.outputLayer, -1, new string[] { "RowSumWeights" });
        }

        /// <summary>
        /// Initializes the structure.
        /// </summary>
        public void InitializeStructure()
        {
            this.Structure = new int[9, 9];
            this.Structure[4, 4] = 1;
        }

        /// <summary>
        /// Reinitialize the structure.
        /// </summary>
        /// <param name="structure">The structure.</param>
        public void ReinitializeAndUpdate(int[,] structure)
        {
            for (int i = 0; i < 9; ++i)
            {
                for (int j = 0; j < 9; ++j)
                {
                    this.Structure[i, j] = structure[i, j];
                }
            }

            this.SetWeightsFromStructure();
        }

        private void SetIndices()
        {
            var startRow = -1;
            var startCol = -1;
            var endRow = -1;
            var endCol = -1;
            var m = 0;
            List<int> ii = new List<int>();
            for (int a = 0; a < 9; ++a)
            {
                for (int b = 0; b < 9; ++b)
                {
                    if (this.Structure[a, b] > 0)
                    {
                        if (startRow == -1)
                        {
                            startRow = a;
                        }

                        if (startCol == -1)
                        {
                            startCol = b;
                        }

                        endRow = a;
                        endCol = b;
                        ii.Add(m);
                    }

                    m++;
                }
            }

            var numRows = endRow - startRow + 1;
            var numCols = endCol - startCol + 1;
            var indices = ii.ToArray();
            this.NumRows = numRows;
            this.NumCols = numCols;
            this.Indices = indices;
        }

        private void UpdateLayerGradients(IModelLayer layer, int layerIndex, string[] identifiers)
        {
            foreach (var identifier in identifiers)
            {
                Matrix[] matrices = layerIndex == -1 ? this[identifier] : this.Layers[layerIndex][identifier];
                int numRows = this.NumRows;
                int numCols = this.NumCols;
                int[] indices = this.Indices;
                int k = 0;

                for (int i = 0; i < numRows; ++i)
                {
                    for (int j = 0; j < numCols; ++j)
                    {
                        var index = indices[k++];

                        var gradients = layer.GradientDeepMatrix(identifier)[index];
                        var subGradients = this.GetSubMatrix(matrices[1], i * gradients.Rows, j * gradients.Cols, gradients.Rows, gradients.Cols);
                        gradients.Accumulate(subGradients.ToArray());
                    }
                }
            }
        }

        private void UpdateLayerGradientsVectors(IModelLayer layer, int layerIndex, string[] identifiers)
        {
            foreach (var identifier in identifiers)
            {
                Matrix[] matrices = layerIndex == -1 ? this[identifier] : this.Layers[layerIndex][identifier];
                int numRows = this.NumRows;
                int numCols = this.NumCols;
                if (numRows > numCols)
                {
                    numRows = numCols;
                }

                int[] indices = this.Indices;
                int k = 0;

                for (int i = 0; i < numRows; ++i)
                {
                    for (int j = 0; j < numCols; ++j)
                    {
                        var index = indices[k++];

                        var gradients = layer.GradientDeepMatrix(identifier)[index];
                        var subGradients = this.GetSubMatrix(matrices[1], i * gradients.Rows, j * gradients.Cols, gradients.Rows, gradients.Cols);
                        gradients.Accumulate(subGradients.ToArray());
                    }
                }
            }
        }

        private void UpdateLayerGradientsBias(IModelLayer layer, int layerIndex, string[] identifiers)
        {
            foreach (var identifier in identifiers)
            {
                Matrix[] matrices = layerIndex == -1 ? this[identifier] : this.Layers[layerIndex][identifier];
                int numRows = 1;
                int numCols = this.NumCols;
                int[] indices = this.Indices;
                int k = 0;

                for (int i = 0; i < numRows; ++i)
                {
                    for (int j = 0; j < numCols; ++j)
                    {
                        var index = indices[k++];

                        var gradients = layer.GradientDeepMatrix(identifier)[index];
                        var subGradients = this.GetSubMatrix(matrices[1], i * gradients.Rows, j * gradients.Cols, gradients.Rows, gradients.Cols);
                        gradients.Accumulate(subGradients.ToArray());
                    }
                }
            }
        }

        private void UpdateLayerGradientsSquare(IModelLayer layer, int layerIndex, string[] identifiers)
        {
            foreach (var identifier in identifiers)
            {
                Matrix[] matrices = layerIndex == -1 ? this[identifier] : this.Layers[layerIndex][identifier];
                int numRows = this.NumRows;
                int numCols = this.NumCols;
                int[] indices = this.Indices;
                if (indices.Length < numCols * numCols)
                {
                    if (numCols == 2)
                    {
                        var ii = new int[] { indices[0], indices[1], indices[0] + 7, indices[1] + 7 };
                        indices = ii;
                    }
                    else if (numCols == 3)
                    {
                        var ii = new int[] { indices[0], indices[1], indices[2], indices[0] + 7, indices[1] + 7, indices[2] + 7, indices[0] + 14, indices[1] + 14, indices[2] + 14 };
                        indices = ii;
                    }
                    else if (numCols == 4)
                    {
                        var ii = new int[] { indices[0], indices[1], indices[2], indices[3], indices[0] + 7, indices[1] + 7, indices[2] + 7, indices[3] + 7, indices[0] + 14, indices[1] + 14, indices[2] + 14, indices[3] + 14, indices[0] + 21, indices[1] + 21, indices[2] + 21, indices[3] + 21 };
                        indices = ii;
                    }
                    else if (numCols == 5)
                    {
                        var ii = new int[] { indices[0], indices[1], indices[2], indices[3], indices[4], indices[0] + 7, indices[1] + 7, indices[2] + 7, indices[3] + 7, indices[4] + 7, indices[0] + 14, indices[1] + 14, indices[2] + 14, indices[3] + 14, indices[4] + 14, indices[0] + 21, indices[1] + 21, indices[2] + 21, indices[3] + 21, indices[4] + 21, indices[0] + 28, indices[1] + 28, indices[2] + 28, indices[3] + 28, indices[4] + 28 };
                        indices = ii;
                    }
                }

                int k = 0;

                for (int i = 0; i < numCols; ++i)
                {
                    for (int j = 0; j < numCols; ++j)
                    {
                        var index = indices[k++];

                        var gradients = layer.GradientDeepMatrix(identifier)[index];
                        var subGradients = this.GetSubMatrix(matrices[1], i * gradients.Rows, j * gradients.Cols, gradients.Rows, gradients.Cols);
                        gradients.Accumulate(subGradients.ToArray());
                    }
                }
            }
        }

        private Matrix[] InitializeWeights(IModelLayer layer, string identifier)
        {
            int numRows = this.NumRows;
            int numCols = this.NumCols;
            int[] indices = this.Indices;
            var mWeights = new Matrix[2];
            var weights = layer.WeightDeepMatrix(identifier)[0];
            var weightRows = numRows * weights.Rows;
            var weightCols = numCols * weights.Cols;

            mWeights[0] = new Matrix(weightRows, weightCols);
            mWeights[1] = new Matrix(weightRows, weightCols);
            int k = 0;
            for (int i = 0; i < numRows; ++i)
            {
                for (int j = 0; j < numCols; ++j)
                {
                    var index = indices[k++];
                    this.SetSubMatrix(mWeights[0], layer.WeightDeepMatrix(identifier)[index], i * weights.Rows, j * weights.Cols);
                }
            }

            return mWeights;
        }

        private Matrix[] InitializeWeightsVectors(IModelLayer layer, string identifier)
        {
            int numRows = this.NumRows;
            int numCols = this.NumCols;
            int[] indices = this.Indices;
            var mWeights = new Matrix[2];
            var weights = layer.WeightDeepMatrix(identifier)[0];
            var weightRows = numRows * weights.Rows;
            var weightCols = numCols * weights.Cols;
            if (weightRows > weightCols)
            {
                weightRows = weightCols;
                numRows = numCols;
            }

            mWeights[0] = new Matrix(weightRows, weightCols);
            mWeights[1] = new Matrix(weightRows, weightCols);
            int k = 0;
            for (int i = 0; i < numRows; ++i)
            {
                for (int j = 0; j < numCols; ++j)
                {
                    var index = indices[k++];
                    this.SetSubMatrix(mWeights[0], layer.WeightDeepMatrix(identifier)[index], i * weights.Rows, j * weights.Cols);
                }
            }

            return mWeights;
        }

        private Matrix[] InitializeWeightsBias(IModelLayer layer, string identifier)
        {
            int numRows = 1;
            int numCols = this.NumCols;
            int[] indices = this.Indices;
            var mWeights = new Matrix[2];
            var weights = layer.WeightDeepMatrix(identifier)[0];
            mWeights[0] = new Matrix(numRows * weights.Rows, numCols * weights.Cols);
            mWeights[1] = new Matrix(numRows * weights.Rows, numCols * weights.Cols);
            int k = 0;
            for (int i = 0; i < numRows; ++i)
            {
                for (int j = 0; j < numCols; ++j)
                {
                    var index = indices[k++];
                    this.SetSubMatrix(mWeights[0], layer.WeightDeepMatrix(identifier)[index], i * weights.Rows, j * weights.Cols);
                }
            }

            return mWeights;
        }

        private Matrix[] InitializeWeightsSquare(IModelLayer layer, string identifier)
        {
            int numRows = this.NumRows;
            int numCols = this.NumCols;
            int[] indices = this.Indices;
            if (indices.Length < numCols * numCols)
            {
                if (numCols == 2)
                {
                    var ii = new int[] { indices[0], indices[1], indices[0] + 7, indices[1] + 7 };
                    indices = ii;
                }
                else if (numCols == 3)
                {
                    var ii = new int[] { indices[0], indices[1], indices[2], indices[0] + 7, indices[1] + 7, indices[2] + 7, indices[0] + 14, indices[1] + 14, indices[2] + 14 };
                    indices = ii;
                }
                else if (numCols == 4)
                {
                    var ii = new int[] { indices[0], indices[1], indices[2], indices[3], indices[0] + 7, indices[1] + 7, indices[2] + 7, indices[3] + 7, indices[0] + 14, indices[1] + 14, indices[2] + 14, indices[3] + 14, indices[0] + 21, indices[1] + 21, indices[2] + 21, indices[3] + 21 };
                    indices = ii;
                }
                else if (numCols == 5)
                {
                    var ii = new int[] { indices[0], indices[1], indices[2], indices[3], indices[4], indices[0] + 7, indices[1] + 7, indices[2] + 7, indices[3] + 7, indices[4] + 7, indices[0] + 14, indices[1] + 14, indices[2] + 14, indices[3] + 14, indices[4] + 14, indices[0] + 21, indices[1] + 21, indices[2] + 21, indices[3] + 21, indices[4] + 21, indices[0] + 28, indices[1] + 28, indices[2] + 28, indices[3] + 28, indices[4] + 28 };
                    indices = ii;
                }
            }

            var mWeights = new Matrix[2];
            var weights = layer.WeightDeepMatrix(identifier)[0];
            mWeights[0] = new Matrix(numCols * weights.Rows, numCols * weights.Cols);
            mWeights[1] = new Matrix(numCols * weights.Rows, numCols * weights.Cols);
            int k = 0;
            for (int i = 0; i < numCols; ++i)
            {
                for (int j = 0; j < numCols; ++j)
                {
                    var index = indices[k++];
                    this.SetSubMatrix(mWeights[0], layer.WeightDeepMatrix(identifier)[index], i * weights.Rows, j * weights.Cols);
                }
            }

            return mWeights;
        }

        private void SetSubMatrix(Matrix contiguous, Matrix subset, int row, int col)
        {
            for (int i = 0; i < subset.Rows; ++i)
            {
                for (int j = 0; j < subset.Cols; ++j)
                {
                    contiguous[row + i, col + j] = subset[i, j];
                }
            }
        }

        private Matrix GetSubMatrix(Matrix contiguous, int row, int col, int numRows, int numCols)
        {
            var subMatrix = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; ++i)
            {
                for (int j = 0; j < numCols; ++j)
                {
                    subMatrix[i, j] = contiguous[row + i, col + j];
                }
            }

            return subMatrix;
        }
    }
}

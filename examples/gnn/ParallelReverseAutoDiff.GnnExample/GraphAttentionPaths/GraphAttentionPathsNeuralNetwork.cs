//------------------------------------------------------------------------------
// <copyright file="GraphAttentionPathsNeuralNetwork.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using System;
    using System.IO;
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.AttentionMessagePassing;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.EdgeAttention;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.Embedding;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.Transformer;

    /// <summary>
    /// Graph Attention Paths Neural Network.
    /// </summary>
    public class GraphAttentionPathsNeuralNetwork
    {
        private const string WEIGHTSSAVEPATH = "D:\\models\\initialWeights2.json";
        private readonly List<EmbeddingNeuralNetwork> embeddingNeuralNetwork;
        private readonly List<EdgeAttentionNeuralNetwork> edgeAttentionNeuralNetwork;
        private readonly List<TransformerNeuralNetwork> transformerNeuralNetwork;
        private readonly List<AttentionMessagePassingNeuralNetwork> attentionMessagePassingNeuralNetwork;
        private readonly GcnNeuralNetwork gcnNeuralNetwork;
        private readonly ReadoutNeuralNetwork readoutNeuralNetwork;
        private readonly List<GapGraph> gapGraphs = new List<GapGraph>();
        private readonly int numFeatures;
        private readonly int numIndices;
        private readonly int numLayers;
        private readonly int numQueries;
        private readonly int alphabetSize;
        private readonly int embeddingSize;
        private readonly double learningRate;
        private readonly double clipValue;
        private readonly Dictionary<int, Guid> typeToIdMap;
        private readonly Dictionary<int, Guid> typeToIdMapTransformer;
        private readonly Dictionary<int, Guid> typeToIdMapAttention;
        private readonly Dictionary<int, Guid> typeToIdMapEmbeddings;
        private readonly Dictionary<GapPath, List<GapPath>> connectedPathsMap;

        private List<IModelLayer> modelLayers;

        /// <summary>
        /// Initializes a new instance of the <see cref="GraphAttentionPathsNeuralNetwork"/> class.
        /// </summary>
        /// <param name="graphs">The graphs.</param>
        /// <param name="numIndices">The number of indices.</param>
        /// <param name="alphabetSize">The alphabet size.</param>
        /// <param name="embeddingSize">The embedding size.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numQueries">The number of queries.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip Value.</param>
        public GraphAttentionPathsNeuralNetwork(List<GapGraph> graphs, int numIndices, int alphabetSize, int embeddingSize, int numLayers, int numQueries, double learningRate, double clipValue)
        {
            this.gapGraphs = graphs;
            this.numFeatures = (numIndices * embeddingSize) + 3;
            this.alphabetSize = alphabetSize;
            this.embeddingSize = embeddingSize;
            this.numIndices = numIndices;
            this.numLayers = numLayers;
            this.numQueries = numQueries;
            this.learningRate = learningRate;
            this.clipValue = clipValue;
            this.modelLayers = new List<IModelLayer>();
            this.embeddingNeuralNetwork = new List<EmbeddingNeuralNetwork>();
            this.edgeAttentionNeuralNetwork = new List<EdgeAttentionNeuralNetwork>();
            this.transformerNeuralNetwork = new List<TransformerNeuralNetwork>();
            this.attentionMessagePassingNeuralNetwork = new List<AttentionMessagePassingNeuralNetwork>();
            this.gcnNeuralNetwork = new GcnNeuralNetwork(numLayers, 4, this.numFeatures, learningRate, clipValue);
            this.readoutNeuralNetwork = new ReadoutNeuralNetwork(numLayers, numQueries, 4, this.numFeatures, learningRate, clipValue);
            this.typeToIdMap = new Dictionary<int, Guid>();
            this.typeToIdMapTransformer = new Dictionary<int, Guid>();
            this.typeToIdMapAttention = new Dictionary<int, Guid>();
            this.typeToIdMapEmbeddings = new Dictionary<int, Guid>();
            this.connectedPathsMap = new Dictionary<GapPath, List<GapPath>>();
        }

        /// <summary>
        /// Reset the network.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task Reset()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.modelLayers.ToArray());

            this.typeToIdMap.Clear();
            this.typeToIdMapTransformer.Clear();
            this.typeToIdMapAttention.Clear();
            this.typeToIdMapEmbeddings.Clear();
            this.connectedPathsMap.Clear();

            for (int i = 0; i < 7; ++i)
            {
                await this.embeddingNeuralNetwork[i].Initialize();
                this.embeddingNeuralNetwork[i].Parameters.AdamIteration++;
            }

            for (int i = 0; i < 7; ++i)
            {
                await this.edgeAttentionNeuralNetwork[i].Initialize();
                this.edgeAttentionNeuralNetwork[i].Parameters.AdamIteration++;
            }

            for (int i = 0; i < 7; ++i)
            {
                await this.transformerNeuralNetwork[i].Initialize();
                this.transformerNeuralNetwork[i].Parameters.AdamIteration++;
            }

            for (int i = 0; i < 7; ++i)
            {
                await this.attentionMessagePassingNeuralNetwork[i].Initialize();
                this.attentionMessagePassingNeuralNetwork[i].Parameters.AdamIteration++;
            }

            await this.gcnNeuralNetwork.Initialize();
            this.gcnNeuralNetwork.Parameters.AdamIteration++;

            await this.readoutNeuralNetwork.Initialize();
            this.readoutNeuralNetwork.Parameters.AdamIteration++;

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
        }

        /// <summary>
        /// Reinitialize with new graphs.
        /// </summary>
        /// <param name="graphs">The graphs.</param>
        public void Reinitialize(List<GapGraph> graphs)
        {
            this.gapGraphs.Clear();
            this.gapGraphs.AddRange(graphs);
        }

        /// <summary>
        /// Initializes the model layers.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            var initialAdamIteration = 633;
            for (int i = 0; i < 7; ++i)
            {
                var model = new EmbeddingNeuralNetwork(this.numIndices, this.alphabetSize, this.embeddingSize, this.learningRate, this.clipValue);
                model.Parameters.AdamIteration = initialAdamIteration;
                this.embeddingNeuralNetwork.Add(model);
                await this.embeddingNeuralNetwork[i].Initialize();
                this.modelLayers = this.modelLayers.Concat(this.embeddingNeuralNetwork[i].ModelLayers).ToList();
            }

            for (int i = 0; i < 7; ++i)
            {
                var model = new EdgeAttentionNeuralNetwork(this.numLayers, this.numQueries, 4, this.numFeatures, this.learningRate, this.clipValue);
                model.Parameters.AdamIteration = initialAdamIteration;
                this.edgeAttentionNeuralNetwork.Add(model);
                await this.edgeAttentionNeuralNetwork[i].Initialize();
                this.modelLayers = this.modelLayers.Concat(this.edgeAttentionNeuralNetwork[i].ModelLayers).ToList();
            }

            for (int i = 0; i < 7; ++i)
            {
                var model = new TransformerNeuralNetwork(this.numLayers, this.numQueries / 2, i + 2, this.numFeatures * (int)Math.Pow(2d, (double)this.numLayers), i + 2, this.learningRate, this.clipValue);
                model.Parameters.AdamIteration = initialAdamIteration;
                this.transformerNeuralNetwork.Add(model);
                await this.transformerNeuralNetwork[i].Initialize();
                this.modelLayers = this.modelLayers.Concat(this.transformerNeuralNetwork[i].ModelLayers).ToList();
            }

            for (int i = 0; i < 7; ++i)
            {
                var model = new AttentionMessagePassingNeuralNetwork(this.numLayers, 4, this.numFeatures, this.learningRate, this.clipValue);
                model.Parameters.AdamIteration = initialAdamIteration;
                this.attentionMessagePassingNeuralNetwork.Add(model);
                await this.attentionMessagePassingNeuralNetwork[i].Initialize();
                this.modelLayers = this.modelLayers.Concat(this.attentionMessagePassingNeuralNetwork[i].ModelLayers).ToList();
            }

            this.gcnNeuralNetwork.Parameters.AdamIteration = initialAdamIteration;
            await this.gcnNeuralNetwork.Initialize();
            this.modelLayers = this.modelLayers.Concat(this.gcnNeuralNetwork.ModelLayers).ToList();

            this.readoutNeuralNetwork.Parameters.AdamIteration = initialAdamIteration;
            await this.readoutNeuralNetwork.Initialize();
            this.modelLayers = this.modelLayers.Concat(this.readoutNeuralNetwork.ModelLayers).ToList();

            // this.SaveWeights();
            // this.ApplyWeights();
        }

        /// <summary>
        /// Save the weights to the save path.
        /// </summary>
        public void SaveWeights()
        {
            Guid guid = Guid.NewGuid();
            var dir = $"E:\\store\\{guid}_{this.readoutNeuralNetwork.Parameters.AdamIteration}";
            Directory.CreateDirectory(dir);
            int index = 0;
            foreach (var modelLayer in this.modelLayers)
            {
                modelLayer.SaveWeightsAndMomentsBinary(new FileInfo($"{dir}\\layer{index}"));
                index++;
            }
        }

        /// <summary>
        /// Apply the weights from the save path.
        /// </summary>
        public void ApplyWeights()
        {
            var guid = "23c09054-6103-4e31-aa03-df30a5eccafc_633";
            var dir = $"E:\\store\\{guid}";
            for (int i = 0; i < this.modelLayers.Count; ++i)
            {
                var modelLayer = this.modelLayers[i];
                var file = new FileInfo($"{dir}\\layer{i}");
                modelLayer.LoadWeightsAndMomentsBinary(file);
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
            }
        }

        /// <summary>
        /// Apply the gradients to update the weights.
        /// </summary>
        public void ApplyGradients()
        {
            var clipper = this.readoutNeuralNetwork.Utilities.GradientClipper;
            clipper.Clip(this.modelLayers.ToArray());
            var adamOptimizer = this.readoutNeuralNetwork.Utilities.AdamOptimizer;
            adamOptimizer.Optimize(this.modelLayers.ToArray());
        }

        /// <summary>
        /// Make a forward pass through the computation graph.
        /// </summary>
        /// <returns>The gradient of the loss wrt the output.</returns>
        public DeepMatrix Forward()
        {
            Dictionary<int, List<Matrix>> indicesByType = new Dictionary<int, List<Matrix>>();
            Dictionary<int, List<Matrix>> featuresByType = new Dictionary<int, List<Matrix>>();
            Dictionary<(int Type, int Index), GapEdge> edgeIndexMap = new Dictionary<(int Type, int Index), GapEdge>();
            foreach (var graph in this.gapGraphs)
            {
                foreach (var node in graph.GapNodes.Where(x => x.IsInPath == true))
                {
                    var edges = node.Edges;
                    for (int i = 0; i < edges.Count; ++i)
                    {
                        var edge = edges[i];
                        Matrix indices = new Matrix(edge.FeatureIndices.Count, 1);
                        for (int j = 0; j < edge.FeatureIndices.Count; ++j)
                        {
                            indices[j, 0] = edge.FeatureIndices[j];
                        }

                        Matrix features = new Matrix(1, edge.Features.Count);
                        for (int j = 0; j < edge.Features.Count; ++j)
                        {
                            features[0, j] = edge.Features[j];
                        }

                        var type = (int)node.GapType;
                        if (!indicesByType.ContainsKey(type))
                        {
                            indicesByType[type] = new List<Matrix>();
                        }

                        indicesByType[type].Add(indices);

                        edgeIndexMap[(type, indicesByType[type].Count - 1)] = edge;

                        if (!featuresByType.ContainsKey(type))
                        {
                            featuresByType[type] = new List<Matrix>();
                        }

                        featuresByType[type].Add(features);
                    }
                }
            }

            foreach (var type in indicesByType.Keys)
            {
                var batchedIndices = new DeepMatrix(indicesByType[type].ToArray());
                var embeddingNet = this.embeddingNeuralNetwork[type];
                embeddingNet.Parameters.BatchSize = batchedIndices.Depth;
                embeddingNet.InitializeState();
                embeddingNet.HandPickedFeatures.Replace(featuresByType[type].ToArray());
                embeddingNet.AutomaticForwardPropagate(batchedIndices);
                var id = Guid.NewGuid();
                this.typeToIdMapEmbeddings.Add(type, id);
                embeddingNet.StoreOperationIntermediates(id);
                var output = embeddingNet.Output;
                for (int i = 0; i < output.Depth; ++i)
                {
                    var edge = edgeIndexMap[(type, i)];
                    edge.FeatureVector = output[i];
                }
            }

            Dictionary<int, List<Matrix>> inputsByType = new Dictionary<int, List<Matrix>>();
            Dictionary<(int Type, int Index), GapNode> nodeIndexMap = new Dictionary<(int Type, int Index), GapNode>();

            foreach (var graph in this.gapGraphs)
            {
                foreach (var node in graph.GapNodes.Where(x => x.IsInPath == true))
                {
                    var edgeCount = node.Edges.Count;
                    if (edgeCount == 0)
                    {
                        node.FeatureVector = new Matrix(this.numFeatures * 4, 1);
                        continue;
                    }

                    var input = new Matrix(edgeCount, this.numFeatures);
                    for (int i = 0; i < edgeCount; ++i)
                    {
                        for (int j = 0; j < this.numFeatures; ++j)
                        {
                            input[i][j] = node.Edges[i].FeatureVector[j][0];
                        }
                    }

                    var index = (int)node.GapType;
                    if (!inputsByType.ContainsKey(index))
                    {
                        inputsByType[index] = new List<Matrix>();
                    }

                    inputsByType[index].Add(input);

                    nodeIndexMap[(index, inputsByType[index].Count - 1)] = node;
                }
            }

            foreach (var type in inputsByType.Keys)
            {
                var batchedInput = new DeepMatrix(inputsByType[type].ToArray());
                var edgeAttentionNet = this.edgeAttentionNeuralNetwork[type];
                edgeAttentionNet.Parameters.BatchSize = batchedInput.Depth;
                edgeAttentionNet.InitializeState();
                edgeAttentionNet.AutomaticForwardPropagate(batchedInput);
                var id = Guid.NewGuid();
                this.typeToIdMap.Add(type, id);
                edgeAttentionNet.StoreOperationIntermediates(id);
                var output = edgeAttentionNet.Output;
                for (int i = 0; i < output.Depth; ++i)
                {
                    var node = nodeIndexMap[(type, i)];
                    node.FeatureVector = output[i];
                }
            }

            Dictionary<int, List<Matrix>> inputsByLength = new Dictionary<int, List<Matrix>>();
            Dictionary<(int Length, int Index), GapPath> pathIndexMap = new Dictionary<(int Length, int Index), GapPath>();

            foreach (var graph in this.gapGraphs)
            {
                foreach (var path in graph.GapPaths)
                {
                    var pathLength = path.Nodes.Count;
                    var input = new Matrix(pathLength, this.numFeatures * (int)Math.Pow(2d, (double)this.numLayers));
                    for (int i = 0; i < input.Rows; ++i)
                    {
                        for (int j = 0; j < input.Cols; ++j)
                        {
                            input[i][j] = path.Nodes[i].FeatureVector[j][0];
                        }
                    }

                    if (!inputsByLength.ContainsKey(pathLength))
                    {
                        inputsByLength[pathLength] = new List<Matrix>();
                    }

                    inputsByLength[pathLength].Add(input);

                    pathIndexMap[(pathLength, inputsByLength[pathLength].Count - 1)] = path;
                }
            }

            foreach (var length in inputsByLength.Keys)
            {
                var batchedInput = inputsByLength[length].ToArray(); // Array of DeepMatrix where each DeepMatrix is a timestep for all sequences in the batch
                var transformerNet = this.transformerNeuralNetwork[length - 2]; // Because a path must have a length of at least two
                transformerNet.Parameters.BatchSize = batchedInput.Length;
                transformerNet.InitializeState();
                transformerNet.AutomaticForwardPropagate(new DeepMatrix(batchedInput));
                var id = Guid.NewGuid();
                this.typeToIdMapTransformer.Add(length, id);
                transformerNet.StoreOperationIntermediates(id);
                var output = transformerNet.Output;
                for (int i = 0; i < output.Depth; ++i)
                {
                    var path = pathIndexMap[(length, i)];
                    path.FeatureVector = output[i];
                }
            }

            Dictionary<int, List<(Matrix, DeepMatrix)>> inputsByTypeAttention = new Dictionary<int, List<(Matrix, DeepMatrix)>>();
            Dictionary<(int Type, int Index), GapPath> pathIndexMapAttention = new Dictionary<(int Type, int Index), GapPath>();

            foreach (var graph in this.gapGraphs)
            {
                foreach (var path in graph.GapPaths)
                {
                    var index = (int)path.GapType;
                    var connectedPaths = graph.GapPaths.Where(x => this.IsConnected(path, x, graph.AdjacencyMatrix)).ToList();
                    this.connectedPathsMap.Add(path, connectedPaths);
                    DeepMatrix connectedPathsMatrix = new DeepMatrix(connectedPaths.Count, this.numFeatures, 1);
                    for (int i = 0; i < connectedPaths.Count; ++i)
                    {
                        var connectedPath = connectedPaths[i];
                        connectedPathsMatrix[i] = connectedPath.FeatureVector;
                    }

                    if (!inputsByTypeAttention.ContainsKey(index))
                    {
                        inputsByTypeAttention[index] = new List<(Matrix, DeepMatrix)>();
                    }

                    inputsByTypeAttention[index].Add((path.FeatureVector, connectedPathsMatrix));

                    pathIndexMapAttention[(index, inputsByTypeAttention[index].Count - 1)] = path;
                }
            }

            foreach (var type in inputsByTypeAttention.Keys)
            {
                var batchedInputs = inputsByTypeAttention[type].Select(x => x.Item1).ToArray();
                var batchedConnectedPaths = inputsByTypeAttention[type].Select(x => x.Item2).ToArray();

                var attentionNet = this.attentionMessagePassingNeuralNetwork[type];
                attentionNet.Parameters.BatchSize = batchedInputs.Length;
                attentionNet.InitializeState();
                attentionNet.ConnectedPathsDeepMatrixArray.Replace(batchedConnectedPaths);
                attentionNet.DConnectedPathsDeepMatrixArray.Replace(batchedConnectedPaths.Select(x => new DeepMatrix(x.Dimension)).ToArray());
                attentionNet.AutomaticForwardPropagate(new DeepMatrix(batchedInputs));
                var id = Guid.NewGuid();
                this.typeToIdMapAttention.Add(type, id);
                attentionNet.StoreOperationIntermediates(id);
                for (int i = 0; i < attentionNet.Output.Depth; ++i)
                {
                    var path = pathIndexMapAttention[(type, i)];
                    path.FeatureVector = attentionNet.Output[i];
                }
            }

            foreach (var graph in this.gapGraphs)
            {
                var gapPaths = graph.GapPaths;
                Matrix adjacency = new Matrix(gapPaths.Count, gapPaths.Count);
                for (int i = 0; i < gapPaths.Count; ++i)
                {
                    var path1 = gapPaths[i];
                    for (int j = 0; j < gapPaths.Count; ++j)
                    {
                        var path2 = gapPaths[j];
                        if (this.IsConnected(path1, path2, graph.AdjacencyMatrix))
                        {
                            var cosineSimilarity = path1.FeatureVector.CosineSimilarity(path2.FeatureVector);
                            var sigmoid = 1.0 / (1.0 + Math.Exp(-cosineSimilarity));
                            adjacency[i][j] = sigmoid;
                        }
                    }
                }

                Matrix degreeMatrix = new Matrix(adjacency.Rows, adjacency.Cols);  // Step 1
                for (int i = 0; i < adjacency.Rows; ++i)
                {
                    int degree = (int)adjacency[i].Sum();
                    degreeMatrix[i, i] = degree != 0 ? Math.Pow(degree, -0.5) : 0;  // Step 2
                }

                Matrix normalizedAdjacency = degreeMatrix * adjacency * degreeMatrix;  // Step 3
                graph.NormalizedAdjacency = normalizedAdjacency;
            }

            List<DeepMatrix> gcnInputList = new List<DeepMatrix>();
            List<Matrix> adjacencyList = new List<Matrix>();

            foreach (var graph in this.gapGraphs)
            {
                // GCN Input
                DeepMatrix gcnInput = new DeepMatrix(graph.GapPaths.Count, this.numFeatures * (int)Math.Pow(2d, (double)this.numLayers) * 2, 1);
                for (int i = 0; i < gcnInput.Depth; ++i)
                {
                    for (int j = 0; j < gcnInput.Rows; ++j)
                    {
                        gcnInput[i][j][0] = graph.GapPaths[i].FeatureVector[j][0];
                    }
                }

                gcnInputList.Add(gcnInput);

                // Normalized Adjacency
                adjacencyList.Add(graph.NormalizedAdjacency);
            }

            var gcnNet = this.gcnNeuralNetwork;
            gcnNet.Parameters.BatchSize = gcnInputList.Count;
            gcnNet.InitializeState();
            gcnNet.Adjacency.Replace(adjacencyList.ToArray());
            gcnNet.AutomaticForwardPropagate(new FourDimensionalMatrix(gcnInputList.ToArray()));
            var gcnOutputs = new DeepMatrix(gcnNet.Output.Last().ToArray());

            var readoutInput = gcnOutputs;
            var readoutNet = this.readoutNeuralNetwork;
            readoutNet.Parameters.BatchSize = readoutInput.Depth;
            readoutNet.InitializeState();
            readoutNet.AutomaticForwardPropagate(readoutInput);
            var readoutOutput = readoutNet.Output;

            List<Matrix> outputGradients = new List<Matrix>();
            for (int i = 0; i < this.gapGraphs.Count; ++i)
            {
                var graph = this.gapGraphs[i];
                var targetFeatures = gcnOutputs[i][0].Length;
                var gapPathTarget = graph.GapPaths.Single(x => x.IsTarget);
                var targetPathIndex = graph.GapPaths.IndexOf(gapPathTarget);
                var targetPath = gcnOutputs[i][targetPathIndex];
                Matrix targetMatrix = new Matrix(targetFeatures, 1);
                for (int j = 0; j < targetFeatures; ++j)
                {
                    targetMatrix[j][0] = targetPath[j];
                }

                CosineDistanceLossOperation cosineDistanceLossOperation = new CosineDistanceLossOperation();
                var loss = cosineDistanceLossOperation.Forward(readoutOutput[i], targetMatrix);

                if (i == 0)
                {
                    List<(GapPath, string, double)> losses = new List<(GapPath, string, double)>();
                    for (int j = 0; j < gcnOutputs[i].Rows; ++j)
                    {
                        var path = gcnOutputs[i][j];
                        var gPath = graph.GapPaths[j];
                        Matrix matrix = new Matrix(path.Length, 1);
                        for (int k = 0; k < path.Length; ++k)
                        {
                            matrix[k][0] = path[k];
                        }

                        CosineDistanceLossOperation operation = new CosineDistanceLossOperation();
                        var result = operation.Forward(readoutOutput[i], matrix);
                        losses.Add((gPath, gPath.Move(), result[0][0]));
                    }

                    var orderedlosses = losses.OrderBy(x => x.Item3).ToList();
                    var avgloss = losses.Average(x => x.Item3);
                    this.PrintGraph(graph, orderedlosses.First().Item2, gapPathTarget.Move(), loss[0][0], avgloss, orderedlosses.First().Item3);
                }

                var gradientOfLossWrtReadoutOutput = cosineDistanceLossOperation.Backward(new Matrix(new[] { new[] { -1.0d } }));
                outputGradients.Add(gradientOfLossWrtReadoutOutput.Item1 as Matrix ?? throw new InvalidOperationException("Gradient should have a value."));
            }

            return new DeepMatrix(outputGradients.ToArray());
        }

        /// <summary>
        /// The backward pass through the computation graph.
        /// </summary>
        /// <param name="gradientOfLossWrtReadoutOutput">The gradient of the loss wrt the output.</param>
        /// <returns>A task.</returns>
        public async Task Backward(DeepMatrix gradientOfLossWrtReadoutOutput)
        {
            var readoutNet = this.readoutNeuralNetwork;
            var inputGradient = await readoutNet.AutomaticBackwardPropagate(gradientOfLossWrtReadoutOutput);
            var gcnNet = this.gcnNeuralNetwork;
            var gcnInputGradient = await gcnNet.AutomaticBackwardPropagate(inputGradient);

            int graphIndex = 0;
            int pathIndex = 0;
            Dictionary<int, List<Matrix>> pathGradients = new Dictionary<int, List<Matrix>>();
            Dictionary<(int, int), GapPath> indexesToPathMap = new Dictionary<(int, int), GapPath>();
            foreach (var graph in this.gapGraphs)
            {
                var gradient = gcnInputGradient[graphIndex];
                pathIndex = 0;
                foreach (var path in graph.GapPaths)
                {
                    var pathGradient = gradient[pathIndex];
                    var index = (int)path.GapType;
                    if (pathGradients.ContainsKey(index))
                    {
                        indexesToPathMap.Add((index, pathGradients[index].Count), path);
                        pathGradients[index].Add(pathGradient);
                    }
                    else
                    {
                        indexesToPathMap.Add((index, 0), path);
                        pathGradients.Add(index, new List<Matrix> { pathGradient });
                    }

                    pathIndex++;
                }

                graphIndex++;
            }

            Dictionary<GapPath, (Matrix, DeepMatrix)> pathToGradientsMap = new Dictionary<GapPath, (Matrix, DeepMatrix)>();
            foreach (var type in pathGradients.Keys)
            {
                var deepMatrixGradient = new DeepMatrix(pathGradients[type].ToArray());
                var attentionNet = this.attentionMessagePassingNeuralNetwork[type];
                attentionNet.RestoreOperationIntermediates(this.typeToIdMapAttention[type]);
                var attentionGradient = await attentionNet.AutomaticBackwardPropagate(deepMatrixGradient);
                var connectedPathsGradient = attentionNet.DConnectedPathsDeepMatrixArray;
                for (int i = 0; i < attentionGradient.Depth; ++i)
                {
                    var path = indexesToPathMap[(type, i)];
                    pathToGradientsMap.Add(path, (attentionGradient[i], connectedPathsGradient[i]));
                }
            }

            this.ApplyGradients(pathToGradientsMap);

            Dictionary<int, List<Matrix>> pathLengthToGradientMap = new Dictionary<int, List<Matrix>>();
            Dictionary<(int, int), GapPath> indexesToPathMapTransformer = new Dictionary<(int, int), GapPath>();
            foreach (var graph in this.gapGraphs)
            {
                foreach (var path in graph.GapPaths)
                {
                    var pathLength = path.Nodes.Count;
                    var gradient = pathToGradientsMap[path].Item1;
                    if (pathLengthToGradientMap.ContainsKey(pathLength))
                    {
                        indexesToPathMapTransformer.Add((pathLength, pathLengthToGradientMap[pathLength].Count), path);
                        pathLengthToGradientMap[pathLength].Add(gradient);
                    }
                    else
                    {
                        indexesToPathMapTransformer.Add((pathLength, 0), path);
                        pathLengthToGradientMap.Add(pathLength, new List<Matrix> { gradient });
                    }
                }
            }

            Dictionary<GapNode, Matrix> nodeToGradientMap = new Dictionary<GapNode, Matrix>();
            foreach (var key in pathLengthToGradientMap.Keys)
            {
                var transformerNet = this.transformerNeuralNetwork[key - 2];
                transformerNet.RestoreOperationIntermediates(this.typeToIdMapTransformer[key]);
                var transformerGradient = await transformerNet.AutomaticBackwardPropagate(new DeepMatrix(pathLengthToGradientMap[key].ToArray()));

                for (int i = 0; i < transformerGradient.Depth; ++i)
                {
                    var path = indexesToPathMapTransformer[(key, i)];
                    var nodeCount = path.Nodes.Count;
                    for (int j = 0; j < nodeCount; ++j)
                    {
                        var node = path.Nodes[j];
                        if (!nodeToGradientMap.ContainsKey(node))
                        {
                            nodeToGradientMap.Add(node, new Matrix(transformerGradient[i][j]).Transpose());
                        }
                        else
                        {
                            nodeToGradientMap[node].Accumulate(new Matrix(transformerGradient[i][j]).Transpose().ToArray());
                        }
                    }
                }
            }

            Dictionary<(int Type, int Index), GapNode> typeIndexNodeMap = new Dictionary<(int Type, int Index), GapNode>();
            Dictionary<int, List<Matrix>> nodeTypeToGradientMap = new Dictionary<int, List<Matrix>>();
            foreach (var graph in this.gapGraphs)
            {
                foreach (var node in graph.GapNodes.Where(x => x.IsInPath == true))
                {
                    if (node.Edges.Count == 0)
                    {
                        continue;
                    }

                    var type = (int)node.GapType;
                    var gradient = nodeToGradientMap[node];
                    if (nodeTypeToGradientMap.ContainsKey(type))
                    {
                        nodeTypeToGradientMap[type].Add(gradient);
                    }
                    else
                    {
                        nodeTypeToGradientMap.Add(type, new List<Matrix> { gradient });
                    }

                    typeIndexNodeMap.Add((type, nodeTypeToGradientMap[type].Count - 1), node);
                }
            }

            Dictionary<GapEdge, Matrix> edgeGradientMap = new Dictionary<GapEdge, Matrix>();
            foreach (var key in nodeTypeToGradientMap.Keys)
            {
                var edgeAttentionNet = this.edgeAttentionNeuralNetwork[key];
                edgeAttentionNet.RestoreOperationIntermediates(this.typeToIdMap[key]);
                var edgeAttentionGradient = await edgeAttentionNet.AutomaticBackwardPropagate(new DeepMatrix(nodeTypeToGradientMap[key].ToArray()));
                for (int i = 0; i < edgeAttentionGradient.Depth; ++i)
                {
                    var node = typeIndexNodeMap[(key, i)];
                    var gradient = edgeAttentionGradient[i];
                    for (int j = 0; j < node.Edges.Count; ++j)
                    {
                        var edge = node.Edges[j];
                        var featureVector = gradient[j];
                        Matrix edgeGradient = new Matrix(featureVector.Length, 1);
                        for (int k = 0; k < featureVector.Length; ++k)
                        {
                            edgeGradient[k][0] = featureVector[k];
                        }

                        edgeGradientMap[edge] = edgeGradient;
                    }
                }
            }

            Dictionary<int, List<GapEdge>> edgesByType = new Dictionary<int, List<GapEdge>>();
            foreach (var graph in this.gapGraphs)
            {
                foreach (var node in graph.GapNodes.Where(x => x.IsInPath == true))
                {
                    var edges = node.Edges;
                    for (int i = 0; i < edges.Count; ++i)
                    {
                        var edge = edges[i];
                        var type = (int)node.GapType;
                        if (!edgesByType.ContainsKey(type))
                        {
                            edgesByType[type] = new List<GapEdge>();
                        }

                        edgesByType[type].Add(edge);
                    }
                }
            }

            foreach (var key in edgesByType.Keys)
            {
                var edges = edgesByType[key];
                DeepMatrix gradients = new DeepMatrix(edges.Select(x => edgeGradientMap[x]).ToArray());
                var embeddingsNet = this.embeddingNeuralNetwork[key];
                embeddingsNet.RestoreOperationIntermediates(this.typeToIdMapEmbeddings[key]);
                var embeddingsGradient = await embeddingsNet.AutomaticBackwardPropagate(gradients);
            }
        }

        private void ApplyGradients(Dictionary<GapPath, (Matrix Gradient, DeepMatrix ConnectedPathsGradient)> pathToGradientsMap)
        {
            foreach (var graph in this.gapGraphs)
            {
                foreach (var path in graph.GapPaths)
                {
                    var (gradient, connectedPathsGradient) = pathToGradientsMap[path];
                    var connectedPaths = this.connectedPathsMap[path];
                    for (int i = 0; i < connectedPaths.Count; ++i)
                    {
                        var connectedPath = connectedPaths[i];
                        var connectedGradient = connectedPathsGradient[i];
                        var grad = pathToGradientsMap[connectedPath].Gradient;
                        grad.Accumulate(connectedGradient.ToArray());
                    }
                }
            }
        }

        private bool IsConnected(GapPath path1, GapPath path2, Matrix adjacency)
        {
            return (int)adjacency[path1.AdjacencyIndex][path2.AdjacencyIndex] == 1;
        }

        private void PrintGraph(GapGraph graph, string move, string target, double targetLoss, double avgloss, double lowestloss)
        {
            // Initialize empty 8x8 board
            string[,] board = new string[8, 8];

            // Fill board spots based on node x/y and piece
            foreach (GapNode node in graph.GapNodes)
            {
                board[node.PositionY, node.PositionX] = node.GapType == GapType.Knight ? "N" : (node.GapType == GapType.Empty ? "." : node.GapType.ToString().Substring(0, 1));
            }

            // Print top border
            Console.WriteLine("  a b c d e f g h");

            // Print board contents
            for (int y = 7; y >= 0; y--)
            {
                Console.Write(" " + (y + 1));
                for (int x = 0; x < 8; x++)
                {
                    Console.Write(board[y, x] + " "); // Print piece or empty
                }

                Console.WriteLine();
            }

            // Print bottom border
            Console.WriteLine("  a b c d e f g h");
            Console.WriteLine("Move: " + move);
            Console.WriteLine("Target:" + target);
            Console.WriteLine("Target Loss: " + targetLoss);
            Console.WriteLine("Avg Loss: " + avgloss);
            Console.WriteLine("Lowest Loss: " + lowestloss);
        }
    }
}

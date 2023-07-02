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
    using ParallelReverseAutoDiff.Test.Common;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.AttentionMessagePassing;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.EdgeAttention;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN;

    /// <summary>
    /// Graph Attention Paths Neural Network.
    /// </summary>
    public class GraphAttentionPathsNeuralNetwork
    {
        private const string WEIGHTSSAVEPATH = "D:\\models\\initialWeights2.json";
        private readonly List<EdgeAttentionNeuralNetwork> edgeAttentionNeuralNetwork;
        private readonly List<LstmNeuralNetwork> lstmNeuralNetwork;
        private readonly List<AttentionMessagePassingNeuralNetwork> attentionMessagePassingNeuralNetwork;
        private readonly GcnNeuralNetwork gcnNeuralNetwork;
        private readonly ReadoutNeuralNetwork readoutNeuralNetwork;
        private readonly List<GapGraph> gapGraphs = new List<GapGraph>();
        private readonly int numFeatures;
        private readonly int numLayers;
        private readonly int numQueries;
        private readonly int batchSize;
        private readonly double learningRate;
        private readonly double clipValue;
        private readonly Dictionary<int, Guid> typeToIdMap;
        private readonly Dictionary<int, Guid> typeToIdMapLstm;
        private readonly Dictionary<int, Guid> typeToIdMapAttention;
        private readonly Dictionary<GapPath, List<GapPath>> connectedPathsMap;

        private List<IModelLayer> modelLayers;

        /// <summary>
        /// Initializes a new instance of the <see cref="GraphAttentionPathsNeuralNetwork"/> class.
        /// </summary>
        /// <param name="graphs">The graphs.</param>
        /// <param name="batchSize">The batch size.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numQueries">The number of queries.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip Value.</param>
        public GraphAttentionPathsNeuralNetwork(List<GapGraph> graphs, int batchSize, int numFeatures, int numLayers, int numQueries, double learningRate, double clipValue)
        {
            this.gapGraphs = graphs;
            this.numFeatures = numFeatures;
            this.numLayers = numLayers;
            this.numQueries = numQueries;
            this.batchSize = batchSize;
            this.learningRate = learningRate;
            this.clipValue = clipValue;
            this.modelLayers = new List<IModelLayer>();
            this.edgeAttentionNeuralNetwork = new List<EdgeAttentionNeuralNetwork>();
            this.typeToIdMap = new Dictionary<int, Guid>();
            this.typeToIdMapLstm = new Dictionary<int, Guid>();
            this.typeToIdMapAttention = new Dictionary<int, Guid>();
            this.connectedPathsMap = new Dictionary<GapPath, List<GapPath>>();
            this.lstmNeuralNetwork = new List<LstmNeuralNetwork>();
            this.attentionMessagePassingNeuralNetwork = new List<AttentionMessagePassingNeuralNetwork>();
            this.gcnNeuralNetwork = new GcnNeuralNetwork(numLayers, 4, numFeatures, learningRate, clipValue);
            this.readoutNeuralNetwork = new ReadoutNeuralNetwork(numLayers, numQueries, 4, numFeatures, learningRate, clipValue);
        }

        /// <summary>
        /// Initializes the model layers.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            for (int i = 0; i < 7; ++i)
            {
                var model = new EdgeAttentionNeuralNetwork(this.numLayers, this.numQueries, 4, this.numFeatures, this.learningRate, this.clipValue);
                this.edgeAttentionNeuralNetwork.Add(model);
                await this.edgeAttentionNeuralNetwork[i].Initialize();
                this.modelLayers = this.modelLayers.Concat(this.edgeAttentionNeuralNetwork[i].ModelLayers).ToList();
            }

            for (int i = 0; i < 7; ++i)
            {
                var model = new LstmNeuralNetwork(this.numFeatures * (int)Math.Pow(2d, (double)this.numLayers), 500, this.numFeatures * (int)Math.Pow(2d, (double)this.numLayers) * 2, i + 2, this.numLayers, this.learningRate, this.clipValue);
                this.lstmNeuralNetwork.Add(model);
                await this.lstmNeuralNetwork[i].Initialize();
                this.modelLayers = this.modelLayers.Concat(this.lstmNeuralNetwork[i].ModelLayers).ToList();
            }

            for (int i = 0; i < 7; ++i)
            {
                var model = new AttentionMessagePassingNeuralNetwork(this.numLayers, 4, this.numFeatures, this.learningRate, this.clipValue);
                this.attentionMessagePassingNeuralNetwork.Add(model);
                await this.attentionMessagePassingNeuralNetwork[i].Initialize();
                this.modelLayers = this.modelLayers.Concat(this.attentionMessagePassingNeuralNetwork[i].ModelLayers).ToList();
            }

            await this.gcnNeuralNetwork.Initialize();
            this.modelLayers = this.modelLayers.Concat(this.gcnNeuralNetwork.ModelLayers).ToList();

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
            var weightStore = new WeightStore();
            weightStore.AddRange(this.modelLayers);
            weightStore.Save(new FileInfo(WEIGHTSSAVEPATH));
        }

        /// <summary>
        /// Apply the weights from the save path.
        /// </summary>
        public void ApplyWeights()
        {
            var weightStore = new WeightStore();
            weightStore.Load(new FileInfo(WEIGHTSSAVEPATH));
            for (int i = 0; i < this.modelLayers.Count; ++i)
            {
                var modelLayer = this.modelLayers[i];
                var weights = weightStore.ToModelLayerWeights(i);
                modelLayer.ApplyWeights(weights);
            }

            weightStore = null;
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced);
        }

        /// <summary>
        /// Make a forward pass through the computation graph.
        /// </summary>
        /// <returns>The gradient of the loss wrt the output.</returns>
        public async Task<DeepMatrix> Forward()
        {
            Dictionary<int, List<Matrix>> inputsByType = new Dictionary<int, List<Matrix>>();
            Dictionary<(int Type, int Index), GapNode> nodeIndexMap = new Dictionary<(int Type, int Index), GapNode>();

            foreach (var graph in this.gapGraphs)
            {
                foreach (var node in graph.GapNodes.Where(x => x.IsInPath == true))
                {
                    var edgeCount = node.Edges.Count;
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

            Dictionary<int, List<DeepMatrix>> inputsByLength = new Dictionary<int, List<DeepMatrix>>();
            Dictionary<(int Length, int Index), GapPath> pathIndexMap = new Dictionary<(int Length, int Index), GapPath>();

            foreach (var graph in this.gapGraphs)
            {
                foreach (var path in graph.GapPaths)
                {
                    var pathLength = path.Nodes.Count;
                    var input = new DeepMatrix(pathLength, this.numFeatures * (int)Math.Pow(2d, (double)this.numLayers), 1);
                    for (int i = 0; i < input.Depth; ++i)
                    {
                        for (int j = 0; j < input.Rows; ++j)
                        {
                            input[i][j][0] = path.Nodes[i].FeatureVector[j][0];
                        }
                    }

                    if (!inputsByLength.ContainsKey(pathLength))
                    {
                        inputsByLength[pathLength] = new List<DeepMatrix>();
                    }

                    inputsByLength[pathLength].Add(input);

                    pathIndexMap[(pathLength, inputsByLength[pathLength].Count - 1)] = path;
                }
            }

            foreach (var length in inputsByLength.Keys)
            {
                var batchedInput = inputsByLength[length].ToArray(); // Array of DeepMatrix where each DeepMatrix is a timestep for all sequences in the batch
                var switched = CommonMatrixUtils.SwitchFirstTwoDimensions(batchedInput);
                var lstmNet = this.lstmNeuralNetwork[length - 2]; // Because a path must have a length of at least two
                lstmNet.Parameters.BatchSize = batchedInput.Length;
                lstmNet.InitializeState();
                await lstmNet.AutomaticForwardPropagate(new FourDimensionalMatrix(switched));
                var id = Guid.NewGuid();
                this.typeToIdMapLstm.Add(length, id);
                lstmNet.StoreOperationIntermediates(id);
                var output = lstmNet.OutputPathFeatures[length - 1];
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
                cosineDistanceLossOperation.Forward(readoutOutput[i], targetMatrix);
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
            Dictionary<(int, int), GapPath> indexesToPathMapLstm = new Dictionary<(int, int), GapPath>();
            foreach (var graph in this.gapGraphs)
            {
                foreach (var path in graph.GapPaths)
                {
                    var pathLength = path.Nodes.Count;
                    var gradient = pathToGradientsMap[path].Item1;
                    if (pathLengthToGradientMap.ContainsKey(pathLength))
                    {
                        indexesToPathMapLstm.Add((pathLength, pathLengthToGradientMap[pathLength].Count), path);
                        pathLengthToGradientMap[pathLength].Add(gradient);
                    }
                    else
                    {
                        indexesToPathMapLstm.Add((pathLength, 0), path);
                        pathLengthToGradientMap.Add(pathLength, new List<Matrix> { gradient });
                    }
                }
            }

            Dictionary<GapNode, Matrix> nodeToGradientMap = new Dictionary<GapNode, Matrix>();
            foreach (var key in pathLengthToGradientMap.Keys)
            {
                var lstmNet = this.lstmNeuralNetwork[key - 2];
                lstmNet.RestoreOperationIntermediates(this.typeToIdMapLstm[key]);
                var lstmGradient = CommonMatrixUtils.SwitchFirstTwoDimensions((await lstmNet.AutomaticBackwardPropagate(new DeepMatrix(pathLengthToGradientMap[key].ToArray()))).ToArray());

                for (int i = 0; i < lstmGradient.Length; ++i)
                {
                    var path = indexesToPathMapLstm[(key, i)];
                    var nodeCount = path.Nodes.Count;
                    for (int j = 0; j < nodeCount; ++j)
                    {
                        var node = path.Nodes[j];
                        if (!nodeToGradientMap.ContainsKey(node))
                        {
                            nodeToGradientMap.Add(node, lstmGradient[i][j]);
                        }
                        else
                        {
                            nodeToGradientMap[node].Accumulate(lstmGradient[i][j].ToArray());
                        }
                    }
                }
            }

            Dictionary<int, List<Matrix>> nodeTypeToGradientMap = new Dictionary<int, List<Matrix>>();
            foreach (var graph in this.gapGraphs)
            {
                foreach (var node in graph.GapNodes.Where(x => x.IsInPath == true))
                {
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
                }
            }

            List<DeepMatrix> edgeGradients = new List<DeepMatrix>();
            foreach (var key in nodeTypeToGradientMap.Keys)
            {
                var edgeAttentionNet = this.edgeAttentionNeuralNetwork[key];
                edgeAttentionNet.RestoreOperationIntermediates(this.typeToIdMap[key]);
                var edgeAttentionGradient = await edgeAttentionNet.AutomaticBackwardPropagate(new DeepMatrix(nodeTypeToGradientMap[key].ToArray()));
                edgeGradients.Add(edgeAttentionGradient);
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
    }
}

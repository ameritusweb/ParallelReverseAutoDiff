namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.Test.Common;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.AttentionMessagePassing;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.EdgeAttention;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN;

    public class GraphAttentionPathsNeuralNetwork
    {
        private const string WEIGHTSSAVEPATH = "C:\\model\\initialWeights3.json";
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
        private readonly Dictionary<GapPath, List<GapPath>> connectedPathsMap;
        private readonly List<IModelLayer> modelLayers;
        private readonly Dictionary<int, Guid> typeToIdMap;
        private readonly Dictionary<int, Guid> typeToIdMapLstm;
        private readonly Dictionary<int, Guid> typeToIdMapAttention;

        public GraphAttentionPathsNeuralNetwork(List<GapGraph> graphs, int batchSize, int numFeatures, int numLayers, int numQueries, double learningRate, double clipValue)
        {
            this.gapGraphs = graphs;
            this.numFeatures = numFeatures;
            this.numLayers = numLayers;
            this.numQueries = numQueries;
            this.batchSize = batchSize;
            this.modelLayers = new List<IModelLayer>();
            this.edgeAttentionNeuralNetwork = new List<EdgeAttentionNeuralNetwork>();
            this.connectedPathsMap = new Dictionary<GapPath, List<GapPath>>();
            this.typeToIdMap = new Dictionary<int, Guid>();
            this.typeToIdMapLstm = new Dictionary<int, Guid>();
            this.typeToIdMapAttention = new Dictionary<int, Guid>();
            for (int i = 0; i < 7; ++i)
            {
                var model = new EdgeAttentionNeuralNetwork(numLayers, numQueries, 4, numFeatures, learningRate, clipValue);
                this.edgeAttentionNeuralNetwork.Add(model);
                this.edgeAttentionNeuralNetwork[i].Initialize();
                this.modelLayers = this.modelLayers.Concat(this.edgeAttentionNeuralNetwork[i].ModelLayers).ToList();
            }

            this.lstmNeuralNetwork = new List<LstmNeuralNetwork>();
            for (int i = 0; i < 7; ++i)
            {
                var model = new LstmNeuralNetwork(numFeatures * (int)Math.Pow(2d, (double)numLayers), 500, numFeatures * (int)Math.Pow(2d, (double)numLayers) * 2, i + 2, numLayers, learningRate, clipValue);
                this.lstmNeuralNetwork.Add(model);
                this.lstmNeuralNetwork[i].Initialize();
                this.modelLayers = this.modelLayers.Concat(this.lstmNeuralNetwork[i].ModelLayers).ToList();
            }

            this.attentionMessagePassingNeuralNetwork = new List<AttentionMessagePassingNeuralNetwork>();
            for (int i = 0; i < 7; ++i)
            {
                var model = new AttentionMessagePassingNeuralNetwork(numLayers, 4, numFeatures, learningRate, clipValue);
                this.attentionMessagePassingNeuralNetwork.Add(model);
                this.attentionMessagePassingNeuralNetwork[i].Initialize();
                this.modelLayers = this.modelLayers.Concat(this.attentionMessagePassingNeuralNetwork[i].ModelLayers).ToList();
            }

            this.gcnNeuralNetwork = new GcnNeuralNetwork(numLayers, 4, numFeatures, learningRate, clipValue);
            this.gcnNeuralNetwork.Initialize();
            this.modelLayers = this.modelLayers.Concat(this.gcnNeuralNetwork.ModelLayers).ToList();

            this.readoutNeuralNetwork = new ReadoutNeuralNetwork(numLayers, numQueries, 4, numFeatures, learningRate, clipValue);
            this.readoutNeuralNetwork.Initialize();
            this.modelLayers = this.modelLayers.Concat(this.readoutNeuralNetwork.ModelLayers).ToList();
            // this.ApplyWeights();
        }

        public void ApplyWeights()
        {
            var weightStore = new WeightStore();    
            weightStore.AddRange(this.modelLayers);
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

        public async Task<DeepMatrix> Forward()
        {
            Dictionary<int, List<Matrix>> inputsByType = new Dictionary<int, List<Matrix>>();
            Dictionary<(int type, int index), GapNode> nodeIndexMap = new Dictionary<(int type, int index), GapNode>();

            foreach (var graph in gapGraphs)
            {
                foreach (var node in graph.GapNodes.Where(x => x.IsInPath == true))
                {
                    var edgeCount = node.Edges.Count;
                    var input = new Matrix(edgeCount, numFeatures);
                    for (int i = 0; i < edgeCount; ++i)
                    {
                        for (int j = 0; j < numFeatures; ++j)
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
            Dictionary<(int length, int index), GapPath> pathIndexMap = new Dictionary<(int length, int index), GapPath>();

            foreach (var graph in gapGraphs)
            {
                foreach (var path in graph.GapPaths)
                {
                    var pathLength = path.Nodes.Count;
                    var input = new DeepMatrix(pathLength, numFeatures * (int)Math.Pow(2d, (double)numLayers), 1);
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
                await lstmNet.AutomaticForwardPropagate(new FourDimensionalMatrix(switched), length);
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
            Dictionary<(int type, int index), GapPath> pathIndexMapAttention = new Dictionary<(int type, int index), GapPath>();

            foreach (var graph in gapGraphs)
            {
                foreach (var path in graph.GapPaths)
                {
                    var index = (int)path.GapType;
                    var connectedPaths = graph.GapPaths.Where(x => IsConnected(path, x, graph.AdjacencyMatrix)).ToList();
                    DeepMatrix connectedPathsMatrix = new DeepMatrix(connectedPaths.Count, numFeatures, 1);
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
                attentionNet.AutomaticForwardPropagate(new DeepMatrix(batchedInputs), true);
                var id = Guid.NewGuid();
                this.typeToIdMapAttention.Add(type, id);
                attentionNet.StoreOperationIntermediates(id);
                for (int i = 0; i < attentionNet.Output.Depth; ++i)
                {
                    var path = pathIndexMapAttention[(type, i)];
                    path.FeatureVector = attentionNet.Output[i];
                }
            }

            foreach (var graph in gapGraphs)
            {
                var gapPaths = graph.GapPaths;
                Matrix adjacency = new Matrix(gapPaths.Count, gapPaths.Count);
                for (int i = 0; i < gapPaths.Count; ++i)
                {
                    var path1 = gapPaths[i];
                    for (int j = 0; j < gapPaths.Count; ++j)
                    {
                        var path2 = gapPaths[j];
                        if (IsConnected(path1, path2, graph.AdjacencyMatrix))
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

            foreach (var graph in gapGraphs)
            {
                // GCN Input
                DeepMatrix gcnInput = new DeepMatrix(graph.GapPaths.Count, numFeatures * (int)Math.Pow(2d, (double)numLayers) * 2, 1);
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
            gcnNet.AutomaticForwardPropagate(new FourDimensionalMatrix(gcnInputList.ToArray()), true);
            var gcnOutputs = new DeepMatrix(gcnNet.Output.Last().ToArray());

            var readoutInput = gcnOutputs;
            var readoutNet = this.readoutNeuralNetwork;
            readoutNet.Parameters.BatchSize = readoutInput.Depth;
            readoutNet.InitializeState();
            readoutNet.AutomaticForwardPropagate(readoutInput, true);
            var readoutOutput = readoutNet.Output;

            List<Matrix> outputGradients = new List<Matrix>();
            for (int i = 0; i < gapGraphs.Count; ++i)
            {
                var graph = gapGraphs[i];
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

        public async Task Backward(DeepMatrix gradientOfLossWrtReadoutOutput)
        {
            var readoutNet = this.readoutNeuralNetwork;
            var inputGradient = await readoutNet.AutomaticBackwardPropagate(gradientOfLossWrtReadoutOutput);
            var gcnNet = this.gcnNeuralNetwork;
            var gcnInputGradient = await gcnNet.AutomaticBackwardPropagate(inputGradient);
            //List<Matrix> attentionNetGradients = new List<Matrix>();
            //List<DeepMatrix> attentionNetConnectedGradients = new List<DeepMatrix>();
            //int pathIndex = 0;
            //foreach (var path in this.gapPaths)
            //{
            //    var index = (int)path.GapType;
            //    var attentionNet = this.attentionMessagePassingNeuralNetwork[index];
            //    attentionNet.RestoreOperationIntermediates(path.Id);
            //    var attentionGradient = await attentionNet.AutomaticBackwardPropagate(gcnInputGradient[pathIndex]);
            //    var connectedPathsGradient = attentionNet.DConnectedPathsDeepMatrix;
            //    attentionNetGradients.Add(attentionGradient);
            //    attentionNetConnectedGradients.Add(connectedPathsGradient);
            //    pathIndex++;
            //}
            //attentionNetGradients = ApplyGradients(attentionNetGradients, attentionNetConnectedGradients);
            //List<DeepMatrix> lstmNetGradients = new List<DeepMatrix>();
            //pathIndex = 0;
            //foreach (var path in this.gapPaths)
            //{
            //    var index = (int)path.GapType;
            //    var lstmNet = this.lstmNeuralNetwork[index];
            //    lstmNet.RestoreOperationIntermediates(path.Id);
            //    var lstmGradient = await lstmNet.AutomaticBackwardPropagate(attentionNetGradients[pathIndex]);
            //    lstmNetGradients.Add(lstmGradient);
            //    pathIndex++;
            //}
            //pathIndex = 0;
            //foreach (var path in this.gapPaths)
            //{
            //    var nodeIndex = 0;
            //    foreach (var node in path.Nodes)
            //    {
            //        var index = (int)node.GapType;
            //        var edgeAttentionNet = this.edgeAttentionNeuralNetwork[index];
            //        edgeAttentionNet.RestoreOperationIntermediates(node.Id);
            //        await edgeAttentionNet.AutomaticBackwardPropagate(lstmNetGradients[pathIndex][nodeIndex]);
            //        nodeIndex++;
            //    }
            //    pathIndex++;
            //}
        }

        //private List<Matrix> ApplyGradients(List<Matrix> attentionNetGradients, List<DeepMatrix> attentionNetConnectedGradients)
        //{
        //    for (int i = 0; i < this.gapPaths.Count; ++i)
        //    {
        //        var path = this.gapPaths[i];
        //        var connectedPaths = this.connectedPathsMap[path];
        //        var gradients = attentionNetConnectedGradients[i];
        //        for (int j = 0; j < connectedPaths.Count; ++j)
        //        {
        //            var connectedPath = connectedPaths[j];
        //            var connectedPathIndex = this.gapPaths.IndexOf(connectedPath);
        //            var gradient = gradients[j];
        //            attentionNetGradients[connectedPathIndex] += gradient;
        //        }   
        //    }
        //    return attentionNetGradients;
        //}

        private bool IsConnected(GapPath path1, GapPath path2, Matrix adjacency)
        {
            return (int)adjacency[path1.AdjacencyIndex][path2.AdjacencyIndex] == 1;
        }
    }
}

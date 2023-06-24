namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using ParallelReverseAutoDiff.RMAD;
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
        private readonly List<GapEdge> gapEdges = new List<GapEdge>();
        private readonly List<GapNode> gapNodes = new List<GapNode>();
        private readonly List<GapPath> gapPaths = new List<GapPath>();
        private readonly int numFeatures;
        private readonly int numLayers;
        private readonly int numQueries;
        private readonly Matrix adjacencyMatrix;
        private readonly Dictionary<GapPath, List<GapPath>> connectedPathsMap;
        private readonly List<IModelLayer> modelLayers;
        private readonly WeightStore weightStore;

        public GraphAttentionPathsNeuralNetwork(List<GapEdge> edges, List<GapNode> nodes, List<GapPath> paths, Matrix adjacencyMatrix, int numFeatures, int numLayers, int numQueries, double learningRate, double clipValue)
        {
            this.gapEdges = edges;
            this.gapNodes = nodes;
            this.gapPaths = paths;
            this.adjacencyMatrix = adjacencyMatrix;
            this.numFeatures = numFeatures;
            this.numLayers = numLayers;
            this.numQueries = numQueries;
            this.modelLayers = new List<IModelLayer>();
            this.weightStore = new WeightStore();
            this.edgeAttentionNeuralNetwork = new List<EdgeAttentionNeuralNetwork>();
            this.connectedPathsMap = new Dictionary<GapPath, List<GapPath>>();
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
                var model = new LstmNeuralNetwork(numFeatures * (int)Math.Pow(2d, (double)numLayers), 500, numFeatures * (int)Math.Pow(2d, (double)numLayers) * 2, 1, numLayers, learningRate, clipValue);
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
            this.weightStore.AddRange(this.modelLayers);
            this.weightStore.Save(new FileInfo(WEIGHTSSAVEPATH));
        }

        public async Task<Matrix> Forward()
        {
            foreach (var node in this.gapNodes.Where(x => x.IsInPath == true))
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
                var edgeAttentionNet = this.edgeAttentionNeuralNetwork[index];
                edgeAttentionNet.AutomaticForwardPropagate(input);
                edgeAttentionNet.StoreOperationIntermediates(node.Id);
                node.FeatureVector = edgeAttentionNet.Output;
            }

            foreach (var path in this.gapPaths)
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
                var index = (int)path.GapType;
                var lstmNet = this.lstmNeuralNetwork[index];
                await lstmNet.AutomaticForwardPropagate(input, pathLength);
                lstmNet.StoreOperationIntermediates(path.Id);
                path.FeatureVector = lstmNet.OutputPathFeatures[pathLength - 1];
            }

            List<Matrix> attentionNetOutputs = new List<Matrix>();
            foreach (var path in this.gapPaths)
            {
                var index = (int)path.GapType;
                var attentionNet = this.attentionMessagePassingNeuralNetwork[index];
                var connectedPaths = this.gapPaths.Where(x => IsConnected(path, x, this.adjacencyMatrix)).ToList();
                DeepMatrix connectedPathsMatrix = new DeepMatrix(connectedPaths.Count, numFeatures, 1);
                for (int i = 0; i < connectedPaths.Count; ++i)
                {
                    var connectedPath = connectedPaths[i];
                    connectedPathsMatrix[i] = connectedPath.FeatureVector;
                }
                this.connectedPathsMap.Add(path, connectedPaths);
                attentionNet.ConnectedPathsDeepMatrix.Replace(connectedPathsMatrix.ToArray());
                attentionNet.DConnectedPathsDeepMatrix.Replace(new DeepMatrix(attentionNet.ConnectedPathsDeepMatrix.Dimension).ToArray());
                attentionNet.AutomaticForwardPropagate(path.FeatureVector, true);
                attentionNet.StoreOperationIntermediates(path.Id);
                attentionNetOutputs.Add(attentionNet.Output);
            }

            for (int i = 0; i < attentionNetOutputs.Count; ++i)
            {
                var path = this.gapPaths[i];
                path.FeatureVector = attentionNetOutputs[i];
            }

            Matrix adjacency = new Matrix(this.gapPaths.Count, this.gapPaths.Count);
            for (int i = 0; i < this.gapPaths.Count; ++i)
            {
                var path1 = this.gapPaths[i];
                for (int j = 0; j < this.gapPaths.Count; ++j)
                {
                    var path2 = this.gapPaths[j];
                    if (IsConnected(path1, path2, this.adjacencyMatrix))
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

            DeepMatrix gcnInput = new DeepMatrix(this.gapPaths.Count, numFeatures * (int)Math.Pow(2d, (double)numLayers) * 2, 1);
            for (int i = 0; i < gcnInput.Depth; ++i)
            {
                for (int j = 0; j < gcnInput.Rows; ++j)
                {
                    gcnInput[i][j][0] = this.gapPaths[i].FeatureVector[j][0];
                }
            }

            var gcnNet = this.gcnNeuralNetwork;
            gcnNet.Adjacency.Replace(normalizedAdjacency.ToArray());
            gcnNet.AutomaticForwardPropagate(gcnInput, true);
            var gcnOutput = gcnNet.Output.Last();

            var readoutInput = gcnOutput;
            var readoutNet = this.readoutNeuralNetwork;
            readoutNet.AutomaticForwardPropagate(readoutInput, true);
            var readoutOutput = readoutNet.Output;

            var targetFeatures = gcnOutput[0].Length;
            var gapPathTarget = this.gapPaths.Single(x => x.IsTarget);
            var targetPathIndex = this.gapPaths.IndexOf(gapPathTarget);
            var targetPath = gcnOutput[targetPathIndex];
            Matrix targetMatrix = new Matrix(targetFeatures, 1);
            for (int i = 0; i < targetFeatures; ++i)
            {
                targetMatrix[i][0] = targetPath[i];
            }

            CosineDistanceLossOperation cosineDistanceLossOperation = new CosineDistanceLossOperation();
            cosineDistanceLossOperation.Forward(readoutOutput, targetMatrix);
            var gradientOfLossWrtReadoutOutput = cosineDistanceLossOperation.Backward(new Matrix(new[] { new[] { -1.0d } }));
            return gradientOfLossWrtReadoutOutput.Item1 as Matrix ?? throw new InvalidOperationException("Gradient should have a value.");
        }

        public async Task Backward(Matrix gradientOfLossWrtReadoutOutput)
        {
            var readoutNet = this.readoutNeuralNetwork;
            var inputGradient = await readoutNet.AutomaticBackwardPropagate(gradientOfLossWrtReadoutOutput);
            var gcnNet = this.gcnNeuralNetwork;
            var gcnInputGradient = await gcnNet.AutomaticBackwardPropagate(inputGradient);
            List<Matrix> attentionNetGradients = new List<Matrix>();
            List<DeepMatrix> attentionNetConnectedGradients = new List<DeepMatrix>();
            int pathIndex = 0;
            foreach (var path in this.gapPaths)
            {
                var index = (int)path.GapType;
                var attentionNet = this.attentionMessagePassingNeuralNetwork[index];
                attentionNet.RestoreOperationIntermediates(path.Id);
                var attentionGradient = await attentionNet.AutomaticBackwardPropagate(gcnInputGradient[pathIndex]);
                var connectedPathsGradient = attentionNet.DConnectedPathsDeepMatrix;
                attentionNetGradients.Add(attentionGradient);
                attentionNetConnectedGradients.Add(connectedPathsGradient);
                pathIndex++;
            }
            attentionNetGradients = ApplyGradients(attentionNetGradients, attentionNetConnectedGradients);
            List<DeepMatrix> lstmNetGradients = new List<DeepMatrix>();
            pathIndex = 0;
            foreach (var path in this.gapPaths)
            {
                var index = (int)path.GapType;
                var lstmNet = this.lstmNeuralNetwork[index];
                lstmNet.RestoreOperationIntermediates(path.Id);
                var lstmGradient = await lstmNet.AutomaticBackwardPropagate(attentionNetGradients[pathIndex]);
                lstmNetGradients.Add(lstmGradient);
                pathIndex++;
            }
            pathIndex = 0;
            foreach (var path in this.gapPaths)
            {
                var nodeIndex = 0;
                foreach (var node in path.Nodes)
                {
                    var index = (int)node.GapType;
                    var edgeAttentionNet = this.edgeAttentionNeuralNetwork[index];
                    edgeAttentionNet.RestoreOperationIntermediates(node.Id);
                    await edgeAttentionNet.AutomaticBackwardPropagate(lstmNetGradients[pathIndex][nodeIndex]);
                    nodeIndex++;
                }
                pathIndex++;
            }
        }

        private List<Matrix> ApplyGradients(List<Matrix> attentionNetGradients, List<DeepMatrix> attentionNetConnectedGradients)
        {
            for (int i = 0; i < this.gapPaths.Count; ++i)
            {
                var path = this.gapPaths[i];
                var connectedPaths = this.connectedPathsMap[path];
                var gradients = attentionNetConnectedGradients[i];
                for (int j = 0; j < connectedPaths.Count; ++j)
                {
                    var connectedPath = connectedPaths[j];
                    var connectedPathIndex = this.gapPaths.IndexOf(connectedPath);
                    var gradient = gradients[j];
                    attentionNetGradients[connectedPathIndex] += gradient;
                }   
            }
            return attentionNetGradients;
        }

        private bool IsConnected(GapPath path1, GapPath path2, Matrix adjacency)
        {
            return (int)adjacency[path1.AdjacencyIndex][path2.AdjacencyIndex] == 1;
        }
    }
}

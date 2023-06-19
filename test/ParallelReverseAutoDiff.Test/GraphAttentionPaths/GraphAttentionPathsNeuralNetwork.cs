namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.AttentionMessagePassing;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.EdgeAttention;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN;

    public class GraphAttentionPathsNeuralNetwork
    {
        private List<EdgeAttentionNeuralNetwork> edgeAttentionNeuralNetwork;
        private List<LstmNeuralNetwork> lstmNeuralNetwork;
        private GcnNeuralNetwork gcnNeuralNetwork;
        private AttentionMessagePassingNeuralNetwork attentionMessagePassingNeuralNetwork;
        private ReadoutNeuralNetwork readoutNeuralNetwork;
        private List<GapEdge> gapEdges;
        private List<GapNode> gapNodes;
        private List<GapPath> gapPaths;
        private int numFeatures;
        private int numLayers;
        private int numQueries;
        private static Random rand = new Random(Guid.NewGuid().GetHashCode());

        public GraphAttentionPathsNeuralNetwork(int numFeatures, int numLayers, int numQueries, double learningRate, double clipValue)
        {
            this.numFeatures = numFeatures;
            this.numLayers = numLayers;
            this.numQueries = numQueries;
            for (int i = 0; i < 8; ++i)
            {
                for (int j = 0; j < 8; j++)
                {
                    var node = new GapNode();
                    node.PositionX = i;
                    node.PositionY = j;
                    node.FeatureVector = new Matrix(numFeatures, 1);
                    node.Edges = new List<GapEdge>();
                    var typeInt = rand.Next() % 7;
                    node.GapType = (GapType)typeInt;
                    this.gapNodes.Add(node);
                    for (int l = 0; l < 5; ++l)
                    {
                        var edge = new GapEdge();
                        edge.Node = node;
                        edge.FeatureVector = new Matrix(numFeatures, 1);
                        edge.FeatureVector.Initialize(InitializationType.Xavier);
                        node.Edges.Add(edge);
                        this.gapEdges.Add(edge);
                    }
                }   
            }
            var path1 = new GapPath();
            path1.FeatureVector = new Matrix(numFeatures, 1);
            path1.Nodes = new List<GapNode>();
            path1.Nodes.Add(this.gapNodes[0]);
            path1.Nodes.Add(this.gapNodes[1]);
            path1.Nodes.Add(this.gapNodes[2]);
            this.gapPaths.Add(path1);

            var path2 = new GapPath();
            path2.FeatureVector = new Matrix(numFeatures, 1);
            path2.Nodes = new List<GapNode>();
            path2.Nodes.Add(this.gapNodes[2]);
            path2.Nodes.Add(this.gapNodes[3]);
            path2.Nodes.Add(this.gapNodes[4]);
            this.gapPaths.Add(path2);

            var path3 = new GapPath();
            path3.FeatureVector = new Matrix(numFeatures, 1);
            path3.Nodes = new List<GapNode>();
            path3.Nodes.Add(this.gapNodes[63]);
            path3.Nodes.Add(this.gapNodes[62]);
            path3.Nodes.Add(this.gapNodes[61]);
            this.gapPaths.Add(path3);

            var path4 = new GapPath();
            path4.FeatureVector = new Matrix(numFeatures, 1);
            path4.Nodes = new List<GapNode>();
            path4.Nodes.Add(this.gapNodes[61]);
            path4.Nodes.Add(this.gapNodes[60]);
            path4.Nodes.Add(this.gapNodes[59]);
            this.gapPaths.Add(path4);

            this.edgeAttentionNeuralNetwork = new List<EdgeAttentionNeuralNetwork>();
            for (int i = 0; i < 7; ++i)
            {
                this.edgeAttentionNeuralNetwork[i] = new EdgeAttentionNeuralNetwork(numLayers, numQueries, 4, numFeatures, learningRate, clipValue);
            }

            this.lstmNeuralNetwork = new List<LstmNeuralNetwork>();
            for (int i = 0; i < 7; ++i)
            {
                this.lstmNeuralNetwork[i] = new LstmNeuralNetwork(numFeatures, 500, numFeatures * 2, 1, numLayers, learningRate, clipValue);
            }

            this.gcnNeuralNetwork = new GcnNeuralNetwork(numLayers, 4, numFeatures, learningRate, clipValue);

            this.attentionMessagePassingNeuralNetwork = new AttentionMessagePassingNeuralNetwork(numLayers, 4, numFeatures, learningRate, clipValue);

            this.readoutNeuralNetwork = new ReadoutNeuralNetwork(numLayers, numQueries, 4, numFeatures, learningRate, clipValue);
        }
    }
}

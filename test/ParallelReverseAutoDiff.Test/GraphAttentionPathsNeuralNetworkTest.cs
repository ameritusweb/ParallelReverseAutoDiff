using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.GraphAttentionPaths;
using Xunit;

namespace ParallelReverseAutoDiff.Test
{
    public class GraphAttentionPathsNeuralNetworkTest
    {
        [Fact]
        public async Task GivenGraphAttentionPathsNeuralNetwork_UsesCudaOperationsSuccessfully()
        {
            CudaBlas.Instance.Initialize();
            try
            {
                Matrix adjacencyMatrix = new Matrix(20, 20);
                for (int i = 0; i < 20; ++i)
                {
                    for (int j = 0; j < 20; ++j)
                    {
                        adjacencyMatrix[i][j] = 1;
                    }
                }
                Random rand = new Random(Guid.NewGuid().GetHashCode());
                List<GapNode> gapNodes = new List<GapNode>();
                List<GapEdge> gapEdges = new List<GapEdge>();   
                List<GapPath> gapPaths = new List<GapPath>();
                int numFeatures = 10;
                for (int i = 0; i < 8; ++i)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        var node = new GapNode();
                        node.Id = Guid.NewGuid();
                        node.PositionX = i;
                        node.PositionY = j;
                        node.FeatureVector = new Matrix(numFeatures, 1);
                        node.Edges = new List<GapEdge>();
                        var typeInt = rand.Next() % 7;
                        node.GapType = (GapType)typeInt;
                        gapNodes.Add(node);
                        for (int l = 0; l < 5; ++l)
                        {
                            var edge = new GapEdge();
                            edge.Node = node;
                            edge.FeatureVector = new Matrix(numFeatures, 1);
                            edge.FeatureVector.Initialize(InitializationType.Xavier);
                            node.Edges.Add(edge);
                            gapEdges.Add(edge);
                        }
                    }
                }
                var path1 = new GapPath();
                path1.IsTarget = true;
                path1.AdjacencyIndex = 0;
                path1.Id = Guid.NewGuid();
                path1.FeatureVector = new Matrix(numFeatures, 1);
                path1.Nodes = new List<GapNode>();
                path1.AddNode(gapNodes[0]);
                path1.AddNode(gapNodes[1]);
                path1.AddNode(gapNodes[2]);
                gapPaths.Add(path1);

                var path2 = new GapPath();
                path2.AdjacencyIndex = 1;
                path2.Id = Guid.NewGuid();
                path2.FeatureVector = new Matrix(numFeatures, 1);
                path2.Nodes = new List<GapNode>();
                path2.AddNode(gapNodes[2]);
                path2.AddNode(gapNodes[3]);
                path2.AddNode(gapNodes[4]);
                gapPaths.Add(path2);

                var path3 = new GapPath();
                path3.Id = Guid.NewGuid();
                path3.AdjacencyIndex = 18;
                path3.FeatureVector = new Matrix(numFeatures, 1);
                path3.Nodes = new List<GapNode>();
                path3.AddNode(gapNodes[63]);
                path3.AddNode(gapNodes[62]);
                path3.AddNode(gapNodes[61]);
                gapPaths.Add(path3);

                var path4 = new GapPath();
                path4.Id = Guid.NewGuid();
                path4.AdjacencyIndex = 19;
                path4.FeatureVector = new Matrix(numFeatures, 1);
                path4.Nodes = new List<GapNode>();
                path4.AddNode(gapNodes[61]);
                path4.AddNode(gapNodes[60]);
                path4.AddNode(gapNodes[59]);
                gapPaths.Add(path4);

                List<GapGraph> graphs = new List<GapGraph>();
                for (int i = 0; i < 8; ++i)
                {
                    graphs[i] = new GapGraph()
                    {
                        AdjacencyMatrix = adjacencyMatrix,
                        GapPaths = gapPaths,
                        GapNodes = gapNodes,
                        GapEdges = gapEdges
                    };
                }

                int batchSize = 8;

                GraphAttentionPathsNeuralNetwork neuralNetwork = new GraphAttentionPathsNeuralNetwork(graphs, batchSize, 10, 2, 4, 0.001d, 4d);
                DeepMatrix gradientOfLoss = await neuralNetwork.Forward();
                await neuralNetwork.Backward(gradientOfLoss);
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }
    }
}
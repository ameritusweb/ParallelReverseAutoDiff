using ParallelReverseAutoDiff.PRAD;
using Xunit;

namespace ParallelReverseAutoDiff.Test.PRAD
{
    public class GraphTests
    {
        [Fact]
        public void BuildGraph()
        {
            var graph = new PradGraph();

            graph.AddNode("Output", 0, 0, 9, 3);
            graph.AddNode("Atan2", 14, 0, 12, 3);
            graph.AddNode("Cos", 31, 0, 14, 3);
            graph.AddNode("Input1", 50, 0, 12, 3);
            graph.AddNode("Sin", 31, 6, 14, 3);
            graph.AddNode("Input2", 50, 6, 13, 3);

            graph.AddEdge("Output", "Atan2");
            graph.AddEdge("Atan2", "Cos");
            graph.AddEdge("Atan2", "Sin");
            graph.AddEdge("Cos", "Input1");
            graph.AddEdge("Sin", "Input2");

            string asciiDiagram = graph.GenerateDiagram();
        }
    }
}

using ParallelReverseAutoDiff.PRAD;
using Xunit;

namespace ParallelReverseAutoDiff.Test.PRAD
{
    public class GraphTests
    {
        [Fact]
        public void BuildGraph()
        {
            var graph = new PradGraph(); // replaces the computation graph

            graph.AddNode("Output", 0, 1, 9, 3);
            graph.AddNode("Mul", 14, 1, 12, 3);
            graph.AddNode("Cos", 31, 1, 14, 3);
            graph.AddNode("Input1", 50, 1, 12, 3);
            graph.AddNode("Sin", 31, 7, 14, 3);
            graph.AddNode("Input2", 50, 7, 13, 3);
            graph.AddNode("Square", 31, 13, 14, 3);
            graph.AddNode("Input3", 50, 13, 13, 3);

            // The PradGraph constructor should take a dictionary that maps a tuple(x, y) of grid placements to actual X and Y coordinates.
            // So for example, (0, 0) would map to (0, 1), and (1, 0) would map to (14, 1), etc.
            // So PradGraph would use this dictionary to place the nodes in the correct positions on the grid.
            // However, I would like to iterate on this and make it even better.
            // So, I would introduce auto-size nodes, where the size of the node is determined by the size of the label.
            // And instead of using a dictionary of grid placements, I would pass in the grid placements and based on the size of the nodes, the actual grid coordinates would be adjusted accordingly.
            // I would have a spacing temperature that I could pass into the PradGraph constructor that would determine the overall spacing between nodes.

            graph.AddEdge("Output", "Mul");
            graph.AddEdge("Mul", "Cos");
            graph.AddEdge("Mul", "Sin");
            graph.AddEdge("Mul", "Square");
            graph.AddEdge("Cos", "Input1");
            graph.AddEdge("Sin", "Input2");
            graph.AddEdge("Square", "Input3");

            string asciiDiagram = graph.GenerateDiagram();
        }
    }
}

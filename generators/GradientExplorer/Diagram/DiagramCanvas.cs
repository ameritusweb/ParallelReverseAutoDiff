using GradientExplorer.Model;
using Microsoft.Msagl.Drawing;
using Microsoft.Msagl.WpfGraphControl;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Diagram
{
    public class DiagramCanvas
    {

        private GradientGraph graph;
        private DiagramViewer viewer;

        public DiagramCanvas(GradientGraph graph)
        {
            this.graph = graph;
            this.viewer = new DiagramViewer();
        }

        public void CreateMsaglGraph(Node gradientNode, Graph msaglGraph, Subgraph parentSubgraph = null)
        {
            // Create MSAGL node if it doesn't exist.
            Microsoft.Msagl.Drawing.Node msaglNode;
            if (!msaglGraph.Nodes.Any(n => n.Id == gradientNode.Id))
            {
                msaglNode = new Microsoft.Msagl.Drawing.Node(gradientNode.Id);
                msaglGraph.AddNode(msaglNode);
            }
            else
            {
                msaglNode = msaglGraph.FindNode(gradientNode.Id);
            }

            // Decide on subgraph creation.
            Subgraph newSubgraph = null;
            if (gradientNode.GradientExpressionType != GradientExpressionType.None)
            {
                newSubgraph = new Subgraph($"cluster_{gradientNode.Id}");
                newSubgraph.AddNode(msaglNode);
                if (parentSubgraph != null)
                {
                    parentSubgraph.AddSubgraph(newSubgraph);
                }
                else
                {
                    msaglGraph.RootSubgraph.AddSubgraph(newSubgraph);
                }
            }

            // Create edges and recurse.
            foreach (var edge in gradientNode.Edges)
            {
                CreateMsaglGraph(edge.TargetNode, msaglGraph, newSubgraph ?? parentSubgraph);
                msaglGraph.AddEdge(gradientNode.Id, edge.TargetNode.Id);
            }
        }


    }
}

using GradientExplorer.Model;
using Microsoft.Msagl.Drawing;
using Microsoft.Msagl.WpfGraphControl;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Media;

namespace GradientExplorer.Diagram
{
    public class DiagramCanvas
    {

        private GradientGraph graph;
        private DiagramViewer viewer;
        private Graph msaglGraph;
        private DockPanel panel;

        public DiagramCanvas(GradientGraph graph)
        {
            this.graph = graph;
            this.msaglGraph = new Graph();
            this.viewer = new DiagramViewer();
            panel = new DockPanel();
            viewer.ObjectUnderMouseCursorChanged += Viewer_ObjectUnderMouseCursorChanged;
            viewer.BindToPanel(panel);
        }

        private void Viewer_ObjectUnderMouseCursorChanged(object sender, ObjectUnderMouseCursorChangedEventArgs e)
        {
            var node = viewer.ObjectUnderMouseCursor as IViewerNode;
            if (node != null)
            {
                var drawingNode = (Microsoft.Msagl.Drawing.Node)node.DrawingObject;
                var text = drawingNode.Label.Text;
            }
            else
            {
                var edge = viewer.ObjectUnderMouseCursor as IViewerEdge;
                if (edge != null)
                {
                    var text = ((Microsoft.Msagl.Drawing.Edge)edge.DrawingObject).SourceNode.Label.Text + "->" +
                                         ((Microsoft.Msagl.Drawing.Edge)edge.DrawingObject).TargetNode.Label.Text;
                }
            }
        }

        public DockPanel ToPanel()
        {
            return panel;
        }

        public void BuildGraph()
        {
            CreateMsaglGraph(this.graph.Nodes.FirstOrDefault());
            viewer.Graph = msaglGraph;
            viewer.GraphCanvas.Width = 300;
            viewer.GraphCanvas.Height = 300;
            ScaleTransform flipTransform = new ScaleTransform(1, -1);
            panel.LayoutTransform = flipTransform;
        }

        private void CreateMsaglGraph(GradientExplorer.Model.Node gradientNode, Subgraph parentSubgraph = null)
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
            if (gradientNode.ExpressionType != GradientExpressionType.None)
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
                CreateMsaglGraph(edge.TargetNode, newSubgraph ?? parentSubgraph);
                msaglGraph.AddEdge(gradientNode.Id, edge.TargetNode.Id);
            }
        }


    }
}

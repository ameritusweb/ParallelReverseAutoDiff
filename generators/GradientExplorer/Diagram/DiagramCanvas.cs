using GradientExplorer.Helpers;
using GradientExplorer.Model;
using GradientExplorer.Services;
using Microsoft.Msagl.Core.Geometry.Curves;
using Microsoft.Msagl.Drawing;
using Microsoft.Msagl.WpfGraphControl;
using Microsoft.VisualStudio.PlatformUI;
using System.Linq;
using System.Windows.Controls;
using System.Windows.Media;

namespace GradientExplorer.Diagram
{
    public class DiagramCanvas
    {

        private GradientGraph graph;
        private DiagramViewer viewer;
        private Graph msaglGraph;
        private Microsoft.Msagl.Drawing.Color backgroundColor;
        private Microsoft.Msagl.Drawing.Color foregroundColor;
        private IEventAggregator eventAggregator;

        public DiagramCanvas(IEventAggregator eventAggregator, GradientGraph graph, Theme theme)
        {
            this.graph = graph;
            this.msaglGraph = new Graph();
            this.viewer = new DiagramViewer();
            this.eventAggregator = eventAggregator;
            this.backgroundColor = theme.MsaglBackgroundColor;
            this.foregroundColor = theme.IsDark ? Microsoft.Msagl.Drawing.Color.White : Microsoft.Msagl.Drawing.Color.Black;
            viewer.ObjectUnderMouseCursorChanged += Viewer_ObjectUnderMouseCursorChanged;
            viewer.GraphCanvas.UpdateLayout();
        }

        public void AddToPanel()
        {
            eventAggregator.PublishAsync(EventType.AddCanvasToPanel, new CanvasEventData { Canvas = viewer.GraphCanvas }).Wait();
        }

        public void Reinitialize(GradientGraph graph, Theme theme)
        {
            this.graph = graph;
            this.msaglGraph = new Graph();
            this.backgroundColor = theme.MsaglBackgroundColor;
            this.foregroundColor = theme.IsDark ? Microsoft.Msagl.Drawing.Color.White : Microsoft.Msagl.Drawing.Color.Black;
            eventAggregator.PublishAsync(EventType.SetPanelLayoutTransform, new PanelLayoutTransformEventData { LayoutTransform = null }).Wait();
            this.viewer.GraphCanvas.UpdateLayout();
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

        public bool UpdateTheme(Theme theme)
        {
            if (this.backgroundColor != theme.MsaglBackgroundColor)
            {
                this.backgroundColor = theme.MsaglBackgroundColor;
                this.foregroundColor = theme.IsDark ? Microsoft.Msagl.Drawing.Color.White : Microsoft.Msagl.Drawing.Color.Black;
                Reinitialize(graph, theme);
                BuildGraph();
                return true;
            }
            return false;
        }

        public void BuildGraph()
        {
            int depth = 0;
            this.msaglGraph.Attr.BackgroundColor = this.backgroundColor;
            this.msaglGraph.Attr.Color = this.foregroundColor;
            CreateMsaglGraph(this.graph.Nodes.FirstOrDefault(), depth);
            viewer.Graph = msaglGraph;
            viewer.GraphCanvas.Width = msaglGraph.Width;
            viewer.GraphCanvas.Height = msaglGraph.Height;
        }

        private void CreateMsaglGraph(GradientExplorer.Model.Node gradientNode, int depth, Subgraph parentSubgraph = null)
        {
            // Create MSAGL node if it doesn't exist.
            Microsoft.Msagl.Drawing.Node msaglNode;
            if (!msaglGraph.Nodes.Any(n => n.Id == gradientNode.Id))
            {
                msaglNode = new Microsoft.Msagl.Drawing.Node(gradientNode.Id);
                msaglNode.Label.Text = gradientNode.DisplayString;
                msaglGraph.AddNode(msaglNode);
                if (parentSubgraph != null)
                {
                    parentSubgraph.AddNode(msaglNode);
                }
            }
            else
            {
                msaglNode = msaglGraph.FindNode(gradientNode.Id);
                msaglNode.Label.Text = gradientNode.DisplayString;
            }

            if (gradientNode.NodeType != NodeType.ConstantOrVariable)
            {
                msaglNode.Attr.Shape = Shape.Diamond;
            }
            else
            {
                msaglNode.Attr.Shape = Shape.Circle;
            }

            if (depth == 0)
            {
                msaglNode.Attr.Shape = Shape.DoubleCircle;
            }

            msaglNode.Label.FontColor = this.foregroundColor;
            msaglNode.Attr.Color = this.foregroundColor;

            // Decide on subgraph creation.
            Subgraph newSubgraph = null;
            if (gradientNode.ExpressionType != GradientExpressionType.None)
            {
                newSubgraph = new Subgraph(gradientNode.ExpressionType.ToString());
                newSubgraph.Attr.Color = this.foregroundColor;
                newSubgraph.Label.FontColor = this.foregroundColor;
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

            depth++;

            // Create edges and recurse.
            for (int i = 0; i < gradientNode.Edges.Count; ++i)
            { 
                var edge = gradientNode.Edges[i];
                CreateMsaglGraph(edge.TargetNode, depth, newSubgraph ?? parentSubgraph);
                var newEdge = msaglGraph.AddEdge(gradientNode.Id, edge.TargetNode.Id);
                newEdge.Attr.Color = this.foregroundColor;
                newEdge.Label = new Microsoft.Msagl.Drawing.Label($"{i}");
                newEdge.Label.FontColor = this.foregroundColor;
                newEdge.Label.FontSize = 12;
                newEdge.Label.Width = 20;
                newEdge.Label.Height = 20;
            }
        }


    }
}

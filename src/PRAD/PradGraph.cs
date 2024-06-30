//------------------------------------------------------------------------------
// <copyright file="PradGraph.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// A diagram of a PRAD graph.
    /// </summary>
    public class PradGraph
    {
        private readonly List<PradNode> nodes = new List<PradNode>();
        private readonly List<PradEdge> edges = new List<PradEdge>();

        /// <summary>
        /// Adds a node to the diagram.
        /// </summary>
        /// <param name="label">The label.</param>
        /// <param name="x">The X coordinate.</param>
        /// <param name="y">The Y coordinate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public void AddNode(string label, int x, int y, int width, int height)
        {
            this.nodes.Add(new PradNode { Label = label, X = x, Y = y, Width = width, Height = height });
        }

        /// <summary>
        /// Adds an edge to the diagram.
        /// </summary>
        /// <param name="fromNode">The from node.</param>
        /// <param name="toNode">The to node.</param>
        public void AddEdge(string fromNode, string toNode)
        {
            var from = this.nodes.Find(n => n.Label == fromNode);
            var to = this.nodes.Find(n => n.Label == toNode);
            if (from != null && to != null)
            {
                this.edges.Add(new PradEdge { From = from, To = to });
            }
        }

        /// <summary>
        /// Generate diagram.
        /// </summary>
        /// <returns>The ASCII diagram.</returns>
        public string GenerateDiagram()
        {
            int maxX = this.nodes.Max(n => n.X + n.Width);
            int maxY = this.nodes.Max(n => n.Y + n.Height);
            var diagram = new PradDiagram(maxX + 1, maxY + 1);

            foreach (var node in this.nodes)
            {
                diagram.AddBox(node.X, node.Y, node.Width, node.Height, node.Label);
            }

            foreach (var edge in this.edges)
            {
                int x1 = edge.From.X + edge.From.Width;
                int y1 = edge.From.Y + (edge.From.Height / 2);
                int x2 = edge.To.X;
                int y2 = edge.To.Y + (edge.To.Height / 2);
                diagram.AddLine(x1, y1, x2, y2);
            }

            return diagram.ToString();
        }
    }
}

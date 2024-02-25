namespace ParallelReverseAutoDiff.GravNetExample
{
    using System.Drawing;
    
    public class Node
    {
        public double NegativeSpaceValue;

        public bool IsForeground { get; set; }
        public double Intensity { get; set; }
        public double GrayValue { get; set; }
        public Point Point { get; set; }
        public double Distance { get; set; }
        public Node Previous { get; set; }
        public int X => Point.X;
        public int Y => Point.Y;

        public Node()
        {

        }

        public Node(int y, int x, bool isForeground)
        {
            Point = new Point(x, y);
            IsForeground = isForeground;
            if (!isForeground)
            {
                Intensity = 255;
            }
        }

        public Node Clone()
        {
            var node = new Node(Y, X, IsForeground);
            node.Distance = int.MaxValue;
            node.Previous = Previous;
            node.NegativeSpaceValue = NegativeSpaceValue;
            return node;
        }
    }
}

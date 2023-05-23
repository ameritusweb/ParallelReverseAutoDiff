using Chess;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GnnExample.GNN
{
    public class ChessSquareGNNNode : GNNNode
    {
        public Position Position { get; set; }
    }
}

using Chess;

namespace ParallelReverseAutoDiff.GnnExample.GNN
{
    public class ChessPieceGNNNode : ChessSquareGNNNode
    {
        public PieceType PieceType { get; set; }
        public PieceColor PieceColor { get; set; }
    }
}

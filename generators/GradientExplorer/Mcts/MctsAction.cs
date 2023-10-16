using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Mcts
{
    public struct MctsAction
    {
        public ITreeNode TreeNode { get; set; }

        public int Depth { get; set; }
    }
}

using FontAwesome.Sharp;
using GradientExplorer.ViewModels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public interface ISortableItem
    {
        string Name { get; }

        IconImageViewModel IconImage { get; }

        bool IsGhost { get; }
    }
}

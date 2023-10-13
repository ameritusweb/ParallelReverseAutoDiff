using GradientExplorer.ViewModels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class SortableItem : ISortableItem
    {
        public string Name { get; set; }

        public IconImageViewModel IconImage { get; set; }

        public bool IsGhost { get; set; }
    }
}

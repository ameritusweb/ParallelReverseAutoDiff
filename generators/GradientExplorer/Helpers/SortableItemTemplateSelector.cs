using GradientExplorer.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows;

namespace GradientExplorer.Helpers
{
    public class SortableItemTemplateSelector : DataTemplateSelector
    {
        public DataTemplate RegularItemTemplate { get; set; }
        public DataTemplate GhostItemTemplate { get; set; }

        public override DataTemplate SelectTemplate(object item, DependencyObject container)
        {
            if (item is SortableItem sortableItem)
            {
                return sortableItem.IsGhost ? GhostItemTemplate : RegularItemTemplate;
            }
            return base.SelectTemplate(item, container);
        }
    }
}

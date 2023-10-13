using GradientExplorer.Model;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.ViewModels
{
    public class ComputationTabViewModel : INotifyPropertyChanged
    {

        private ObservableCollection<ISortableItem> _computationGraph;

        public ObservableCollection<ISortableItem> ComputationGraph
        {
            get { return _computationGraph; }
            set
            {
                if (_computationGraph != value)
                {
                    _computationGraph = value;
                    OnPropertyChanged(nameof(ComputationGraph));
                }
            }
        }

        public ComputationTabViewModel(GradientExplorerViewModel parent) {

            this.Parent = parent;

            ComputationGraph = new ObservableCollection<ISortableItem>
            {
                new SortableItem 
                { 
                    Name = "Item 1", 
                    IconImage = new IconImageViewModel { 
                        Foreground = System.Windows.Media.Brushes.CornflowerBlue,
                        Height = 20,
                        Icon = FontAwesome.Sharp.IconChar.Android,
                    },  
                    IsGhost = false 
                },
                new SortableItem
                {
                    Name = "Item 2",
                    IconImage = new IconImageViewModel {
                        Foreground = System.Windows.Media.Brushes.CornflowerBlue,
                        Height = 20,
                        Icon = FontAwesome.Sharp.IconChar.KissWinkHeart,
                    },
                    IsGhost = false
                },
            };

        }

        public GradientExplorerViewModel Parent { get; set; }

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}

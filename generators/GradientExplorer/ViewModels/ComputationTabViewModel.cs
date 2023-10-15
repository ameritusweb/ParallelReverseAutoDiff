using GradientExplorer.Helpers;
using GradientExplorer.Model;
using GradientExplorer.Services;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace GradientExplorer.ViewModels
{
    public class ComputationTabViewModel : IViewModel, INotifyPropertyChanged
    {

        private ILogger logger;

        public ILogger Logger
        {
            get
            {
                return logger;
            }
        }

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

        public ComputationTabViewModel(GradientExplorerViewModel parent, ILogger logger) {

            this.Parent = parent;
            this.logger = logger;

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
                new SortableItem
                {
                    Name = "Item 3",
                    IconImage = new IconImageViewModel {
                        Foreground = System.Windows.Media.Brushes.CornflowerBlue,
                        Height = 20,
                        Icon = FontAwesome.Sharp.IconChar.Amazon,
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

using GradientExplorer.ViewModels;
using System.ComponentModel;

namespace GradientExplorer.Model
{
    public class SortableItem : ISortableItem, INotifyPropertyChanged
    {
        private string _name;
        private IconImageViewModel _iconImage;
        private bool _isGhost;

        public string Name
        {
            get => _name;
            set
            {
                if (_name != value)
                {
                    _name = value;
                    OnPropertyChanged(nameof(Name));
                }
            }
        }

        public IconImageViewModel IconImage
        {
            get => _iconImage;
            set
            {
                if (_iconImage != value)
                {
                    _iconImage = value;
                    OnPropertyChanged(nameof(IconImage));
                }
            }
        }

        public bool IsGhost
        {
            get => _isGhost;
            set
            {
                if (_isGhost != value)
                {
                    _isGhost = value;
                    OnPropertyChanged(nameof(IsGhost));
                }
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}

using System.Windows.Controls;
using System.Windows.Media.Animation;
using System.Windows.Media;
using System.Windows;
using GradientExplorer.Model;
using System.Collections.ObjectModel;
using System.Windows.Input;
using System.Threading.Tasks;
using System.Linq;
using GradientExplorer.Controls;

namespace GradientExplorer.Helpers
{
    public static class SortableListBoxBehavior
    {
        public static readonly DependencyProperty AllowSortProperty =
            DependencyProperty.RegisterAttached("AllowSort", typeof(bool), typeof(SortableListBoxBehavior),
                new PropertyMetadata(false, OnAllowSortChanged));

        public static bool GetAllowSort(DependencyObject d)
        {
            return (bool)d.GetValue(AllowSortProperty);
        }

        public static void SetAllowSort(DependencyObject d, bool value)
        {
            d.SetValue(AllowSortProperty, value);
        }

        public static readonly DependencyProperty AnimationDurationProperty =
            DependencyProperty.RegisterAttached("AnimationDuration", typeof(double), typeof(SortableListBoxBehavior),
                new PropertyMetadata(0.3));

        public static double GetAnimationDuration(DependencyObject d)
        {
            return (double)d.GetValue(AnimationDurationProperty);
        }

        public static void SetAnimationDuration(DependencyObject d, double value)
        {
            d.SetValue(AnimationDurationProperty, value);
        }

        public static readonly DependencyProperty EasingFunctionProperty =
    DependencyProperty.RegisterAttached("EasingFunction", typeof(EasingFunctionBase), typeof(SortableListBoxBehavior),
        new PropertyMetadata(new CubicEase { EasingMode = EasingMode.EaseInOut }));

        public static EasingFunctionBase GetEasingFunction(DependencyObject d)
        {
            return (EasingFunctionBase)d.GetValue(EasingFunctionProperty);
        }

        public static void SetEasingFunction(DependencyObject d, EasingFunctionBase value)
        {
            d.SetValue(EasingFunctionProperty, value);
        }

        private static void OnAllowSortChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is SortableListBox listBox)
            {
                if ((bool)e.NewValue)
                {
                    listBox.PreviewMouseLeftButtonDown += SortableListBox_PreviewMouseLeftButtonDown;
                    listBox.PreviewMouseLeftButtonUp += SortableListBox_PreviewMouseLeftButtonUp;
                    listBox.PreviewMouseMove += SortableListBox_PreviewMouseMove;  // New line
                    listBox.DragOver += SortableListBox_DragOver;
                    listBox.Drop += SortableListBox_Drop;
                }
                else
                {
                    listBox.PreviewMouseLeftButtonDown -= SortableListBox_PreviewMouseLeftButtonDown;
                    listBox.PreviewMouseLeftButtonUp -= SortableListBox_PreviewMouseLeftButtonUp;
                    listBox.PreviewMouseMove -= SortableListBox_PreviewMouseMove;  // New line
                    listBox.DragOver -= SortableListBox_DragOver;
                    listBox.Drop -= SortableListBox_Drop;
                }
            }
        }

        private static void SortableListBox_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (sender is SortableListBox listBox)
            {
                if (e.LeftButton == MouseButtonState.Pressed && listBox.IsPotentialDrag)
                {
                    var hitTestResult = VisualTreeHelper.HitTest(listBox, e.GetPosition(listBox));
                    var listBoxItem = FindVisualParent<ListBoxItem>(hitTestResult.VisualHit);

                    if (listBoxItem != null)
                    {
                        listBoxItem.IsSelected = true;
                        (listBoxItem.DataContext as ISortableItem).IsGhost = true;
                        DragDrop.DoDragDrop(listBoxItem, listBoxItem.DataContext, DragDropEffects.Move);

                        // Reset the flag since we've initiated the drag-and-drop
                        listBox.IsPotentialDrag = false;
                    }
                }
            }
        }

        private static void SortableListBox_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            // Set a flag to indicate potential drag operation
            if (sender is SortableListBox listBox)
            {
                listBox.IsPotentialDrag = true;
            }
        }

        private static void SortableListBox_PreviewMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            // Sender is SortableListBox, so we need to find the ListBoxItem that was initially clicked
            if (sender is SortableListBox listBox)
            {
                // Reset the flag since the mouse button is released
                listBox.IsPotentialDrag = false;
                var hitTestResult = VisualTreeHelper.HitTest(listBox, e.GetPosition(listBox));
                var listBoxItem = FindVisualParent<ListBoxItem>(hitTestResult.VisualHit);

                if (listBoxItem != null)
                {
                    (listBoxItem.DataContext as ISortableItem).IsGhost = false;
                }
            }
        }

        private static T FindVisualParent<T>(DependencyObject child) where T : DependencyObject
        {
            while (child != null && !(child is T))
            {
                child = VisualTreeHelper.GetParent(child);
            }
            return child as T;
        }

        private static async void SortableListBox_DragOver(object sender, DragEventArgs e)
        {
            e.Handled = true;
            SortableListBox listBox = sender as SortableListBox;
            var itemsSource = listBox.ItemsSource as ObservableCollection<ISortableItem>;

            if (itemsSource == null)
            {
                throw new InvalidOperationException("ItemsSource is not an ObservableCollection<Item>");
            }

            Point position = e.GetPosition(listBox);
            int? index = GetCurrentIndex(listBox, position);
            int selectedIndex = listBox.SelectedIndex;

            if (index.HasValue && index != selectedIndex && !listBox.IsAnimating)
            {
                var oldItem = listBox.ItemContainerGenerator.ContainerFromItem(itemsSource[selectedIndex]) as ListBoxItem;
                var newItem = listBox.ItemContainerGenerator.ContainerFromItem(itemsSource[index.Value]) as ListBoxItem;

                if (oldItem == null || newItem == null) return;

                foreach (var item in itemsSource)
                {
                    item.IsGhost = false;
                }

                try
                {

                    listBox.IsAnimating = true;

                    // Set the ghost state for animation purposes
                    itemsSource[selectedIndex].IsGhost = true;

                    // Increase the Z-Index of the dragged item to make it appear above the other item
                    Panel.SetZIndex(oldItem, 1);
                    Panel.SetZIndex(newItem, 0);

                    double oldItemY = oldItem.TranslatePoint(new Point(0, 0), listBox).Y;
                    double newItemY = newItem.TranslatePoint(new Point(0, 0), listBox).Y;

                    // Trigger animations and await their completion
                    var oldStoryboard = AnimateRow(oldItem, newItemY - oldItemY);
                    var newStoryboard = AnimateRow(newItem, oldItemY - newItemY);

                    await Task.WhenAll(oldStoryboard, newStoryboard);

                    // Reset Z-Index to default values
                    Panel.SetZIndex(oldItem, 0);
                    Panel.SetZIndex(newItem, 0);

                    oldItem.RenderTransform = null;
                    newItem.RenderTransform = null;

                    // Perform the swap in data
                    SwapItems(itemsSource, index.Value, selectedIndex);

                    listBox.SelectedIndex = index.Value;
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
                finally
                {
                    listBox.IsAnimating = false;
                }
            }
        }

        private static int? GetCurrentIndex(SortableListBox listBox, Point position, double tolerance = 5.0)
        {
            for (int i = 0; i < listBox.Items.Count; i++)
            {
                ListBoxItem item = (ListBoxItem)listBox.ItemContainerGenerator.ContainerFromIndex(i);
                ISortableItem sortableItem = item.DataContext as ISortableItem;
                if (!item.IsSelected && !sortableItem.IsGhost)
                {
                    if (IsMouseOverTarget(listBox, item, position, tolerance))
                    {
                        return i;
                    }
                }
            }
            return default(int?);
        }

        private static bool IsMouseOverTarget(SortableListBox listBox, Visual target, Point point, double tolerance)
        {
            Rect bounds = VisualTreeHelper.GetDescendantBounds(target);

            // Transform the bounds to the coordinate space of the parent SortableListBox
            GeneralTransform transform = target.TransformToAncestor(listBox);
            bounds = transform.TransformBounds(bounds);
            bounds = new Rect(bounds.Left + tolerance, bounds.Top + tolerance,
                              bounds.Width - 2 * tolerance, bounds.Height - 2 * tolerance);
            var withinBounds = bounds.Contains(point);
            if (withinBounds)
            {
                
            }
            return withinBounds;
        }

        private static void SortableListBox_Drop(object sender, DragEventArgs e)
        {
            SortableListBox listBox = sender as SortableListBox;
            var itemsSource = listBox.ItemsSource as ObservableCollection<ISortableItem>;

            if (itemsSource == null)
            {
                throw new InvalidOperationException("ItemsSource is not an ObservableCollection<Item>");
            }

            // Reset any transformations applied during the drag-and-drop operation
            foreach (var item in itemsSource)
            {
                var listBoxItem = listBox.ItemContainerGenerator.ContainerFromItem(item) as ListBoxItem;
                if (listBoxItem != null)
                {
                    listBoxItem.RenderTransform = null;
                    Panel.SetZIndex(listBoxItem, 0);  // Reset Z-Index to default
                }
            }

            // Find the 'ghost' item and animate opacity back to 1
            var ghostItem = itemsSource.FirstOrDefault(x => x.IsGhost);
            if (ghostItem != null)
            {
                // Remove the ghost state
                ghostItem.IsGhost = false;
            }

            e.Handled = true;
        }

        private static async Task AnimateRow(ListBoxItem item, double translateY)
        {
            var tcs = new TaskCompletionSource<bool>();
            var translateTransform = new TranslateTransform();
            item.RenderTransform = translateTransform;

            var path = new PathGeometry();
            var pf = new PathFigure
            {
                StartPoint = new Point(0, 0),
                IsClosed = false
            };
            var arcSeg = new ArcSegment
            {
                Point = new Point(0, translateY),
                Size = new Size(Math.Abs(translateY), Math.Abs(translateY)),
                IsLargeArc = false,
                SweepDirection = SweepDirection.Counterclockwise
            };
            pf.Segments.Add(arcSeg);
            path.Figures.Add(pf);

            var xAnimation = new DoubleAnimationUsingPath
            {
                PathGeometry = path,
                Source = PathAnimationSource.X,
                Duration = TimeSpan.FromMilliseconds(150),
                FillBehavior = FillBehavior.HoldEnd
            };

            var yAnimation = new DoubleAnimationUsingPath
            {
                PathGeometry = path,
                Source = PathAnimationSource.Y,
                Duration = TimeSpan.FromMilliseconds(150),
                FillBehavior = FillBehavior.HoldEnd
            };

            var xBuilder = new PropertyPathBuilder()
                .WithDependencyProperty(UIElement.RenderTransformProperty)
                .WithDependencyProperty(TranslateTransform.XProperty);

            var yBuilder = new PropertyPathBuilder()
                .WithDependencyProperty(UIElement.RenderTransformProperty)
                .WithDependencyProperty(TranslateTransform.YProperty);

            var storyboard = new Storyboard();
            storyboard.AutoReverse = false;
            storyboard.Children.Add(xAnimation);
            storyboard.Children.Add(yAnimation);
            Storyboard.SetTarget(xAnimation, item);
            Storyboard.SetTarget(yAnimation, item);
            Storyboard.SetTargetProperty(xAnimation, xBuilder.Build());
            Storyboard.SetTargetProperty(yAnimation, yBuilder.Build());

            storyboard.Completed += (s, e) => tcs.SetResult(true);
            storyboard.Begin();

            await tcs.Task;
        }

        private static void SwapItems(ObservableCollection<ISortableItem> itemsSource, int index1, int index2)
        {
            ISortableItem temp = itemsSource[index1];
            itemsSource[index1] = itemsSource[index2];
            itemsSource[index2] = temp;
        }

    }
}

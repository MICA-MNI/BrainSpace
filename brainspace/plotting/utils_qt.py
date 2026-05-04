try:
    from PyQt5.QtCore import pyqtSignal
    from PyQt5.QtWidgets import QMainWindow
except ImportError:
    MainWindow = None
else:
    class MainWindow(QMainWindow):
        """Subclass of QMainWindow to capture closing event."""

        signal_close = pyqtSignal()

        def __init__(self, parent=None):
            """Initialize the main window."""
            super().__init__(parent)

        def closeEvent(self, event):
            """Manage the close event."""
            self.signal_close.emit()
            event.accept()

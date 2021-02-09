"""
Commandline argument parsing and other helper functions
"""

import os
import sys
import glob
import math
import argparse
from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget, QFileDialog
from PyQt5.QtWidgets import QCheckBox, QPushButton, QComboBox, QLabel, QDialog, QListWidget
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QDialogButtonBox, QGridLayout, QLineEdit
from PyQt5.QtCore import Qt

USE_QT_FOR_FILEIO = False

def display_warning(info):
    """
    Show a dialog box with the specified message.
    """
    # print('Opening message box ' + info)
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText(info)
    msg.setWindowTitle('Warning!')
    msg.setStandardButtons(QMessageBox.Ok)
    return msg.exec_()

def display_information(info):
    """
    Show a dialog box with the specified message.
    """
    # print('Opening message box ' + info)
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(info)
    msg.setWindowTitle('Message')
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    return msg.exec_()

class RippleSelectionMenuWidget(QDialog):
    """
    Dialog-box that allows the user to choose a tetrode to be set as the ripple
    reference and baseline.  This can be converted into a more general
    selection tool but not for now.
    """
    def __init__(self, tetrode_list):
        """
        Create the dialog box using the supplied tetrode list.
        """

        QDialog.__init__(self)
        # self.setIcon(QMessageBox.Information)
        # self.setText("Ripple preference menu")
        self.setWindowTitle("Ripple preference menu")

        self.ripple_reference_selection = QComboBox()
        self.ripple_baseline_selection = QComboBox()

        # Add tetrode items to the ripple preferences menu
        tetrode_id_strings = [str(tet_id) for tet_id in tetrode_list]

        self.ripple_reference_selection.addItems(tetrode_id_strings)
        self.ripple_baseline_selection.addItems(tetrode_id_strings)

        # Add the two entries in one line each, appending a label next to them
        reference_layout = QHBoxLayout()
        reference_layout.addStretch(1)
        reference_layout.addWidget(QLabel("Reference"))
        reference_layout.addWidget(self.ripple_reference_selection)

        baseline_layout = QHBoxLayout()
        baseline_layout.addStretch(1)
        baseline_layout.addWidget(QLabel("Baseline"))
        baseline_layout.addWidget(self.ripple_baseline_selection)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        selection_menu_layout = QVBoxLayout()
        selection_menu_layout.addLayout(reference_layout)
        selection_menu_layout.addLayout(baseline_layout)
        selection_menu_layout.addWidget(self.button_box)
        self.setLayout(selection_menu_layout)

        # Hold the indexes selected last by the user as class members
        self._sel_reference_idx = 0
        self._sel_baseline_idx = 0

    def exec_(self, *args, **kwargs):
        """
        Overloading the exec_() function to get both the messagebox result,
        as well as the selected checkboxes.
        """
        self.ripple_reference_selection.setCurrentIndex(self._sel_reference_idx)
        self.ripple_baseline_selection.setCurrentIndex(self._sel_baseline_idx)
        ok = QDialog.exec_(self, *args, *kwargs)
        self._sel_reference_idx = self.ripple_reference_selection.currentIndex()
        self._sel_baseline_idx  = self.ripple_baseline_selection.currentIndex()
        return ok

    def setIdxs(self, r_idx, b_idx):
        if r_idx is not None:
            self._sel_reference_idx = r_idx
        if b_idx is not None:
            self._sel_baseline_idx = b_idx

    def getIdxs(self):
        return self._sel_reference_idx, self._sel_baseline_idx

class BrainCoordinateWidget(QDialog):

    """
    Dialog box to pick the correct brain coordinates and viewing plane.
    """

    def __init__(self, ml=0.0, ap=0.0, dv=0.0):
        QDialog.__init__(self)
        self.setWindowTitle("Select coordinates and viewing plane.")

        # ML data entry
        ml_label = QLabel("ML")
        self.ml_data = QLineEdit(self)
        self.ml_data.setText(str(ml))
        ml_layout = QHBoxLayout()
        ml_layout.addWidget(ml_label)
        ml_layout.addWidget(self.ml_data)

        # AP data entry
        ap_label = QLabel("AP")
        self.ap_data = QLineEdit(self)
        self.ap_data.setText(str(ap))
        ap_layout = QHBoxLayout()
        ap_layout.addWidget(ap_label)
        ap_layout.addWidget(self.ap_data)

        # DV data entry
        dv_label = QLabel("DV")
        self.dv_data = QLineEdit(self)
        self.dv_data.setText(str(dv))
        dv_layout = QHBoxLayout()
        dv_layout.addWidget(dv_label)
        dv_layout.addWidget(self.dv_data)

        # View selection - Checkboxes
        # Also have the coronal view be selected by default
        coronal_view = QCheckBox("Coronal", self)
        coronal_view.setChecked(True)
        sagittal_view = QCheckBox("Sagittal", self)
        horizontal_view = QCheckBox("Horizontal", self)
        self.views = list()
        self.views.append(coronal_view)
        self.views.append(sagittal_view)
        self.views.append(horizontal_view)

        view_selection = QHBoxLayout()
        view_selection.addWidget(coronal_view)
        view_selection.addWidget(sagittal_view)
        view_selection.addWidget(horizontal_view)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        g_layout = QVBoxLayout()
        g_layout.addLayout(ml_layout)
        g_layout.addLayout(ap_layout)
        g_layout.addLayout(dv_layout)
        g_layout.addLayout(view_selection)
        g_layout.addWidget(self.button_box, alignment=Qt.AlignCenter)
        self.setLayout(g_layout)

    def exec_(self, *args, **kwargs):
        """
        Overloading the exec_() function to get both the messagebox result,
        as well as the selected checkboxes.
        """
        ok = QDialog.exec_(self, *args, *kwargs)
        ml_coord = float(self.ml_data.text())
        ap_coord = float(self.ap_data.text())
        dv_coord = float(self.dv_data.text())
        accepted_idxs = list()
        for idx, chk in enumerate(self.views):
            if chk.isChecked():
                accepted_idxs.append(idx)
        return ok, [ml_coord, ap_coord, dv_coord], accepted_idxs

class ListDisplayWidget(QDialog):

    """
    Dialog-box showing a list of numbers.
    """

    def __init__(self, title, arg_id, arg_field1, arg_field2=None):
        QDialog.__init__(self)
        if title is not None:
            self.setWindowTitle(title)
        else:
            self.setWindowTitle('Data')

        self.data_list_widget = QListWidget()
        n_data_entries = len(arg_id)
        self.data_list_widget.resize(100,20*n_data_entries)

        # It is upto the user to supply correctly sized lists in this case.
        for data_idx, data_value in enumerate(arg_id):
            if arg_field2 is None:
                self.data_list_widget.addItem("%s: %.2f"%(data_value, arg_field1[data_idx]))
            else:
                self.data_list_widget.addItem("%s: %.2f, %.2f"%(data_value, arg_field1[data_idx], \
                        arg_field2[data_idx]))

        g_layout = QVBoxLayout()
        g_layout.addWidget(self.data_list_widget)


        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        g_layout.addWidget(self.button_box)
        self.setLayout(g_layout)

class CheckBoxWidget(QDialog):
  
    """
    Dialog-box containing a number of check-boxes.
    """

    MAX_CHECKBOX_COLUMNS = 2
    def __init__(self, list_of_elements, message=None, default_choices=None):
        """
        Create the dialog-box with a pre-defined list of elements.
        """

        QDialog.__init__(self)
        if message is not None:
            self.setWindowTitle(message)
        else:
            self.setWindowTitle('Message')
        self.checkboxes = list()

        if default_choices is None:
            n_elements = len(list_of_elements)
            default_choices = [True for el_idx in range(n_elements)]

        # Set up the layout... All the list elements in a single column
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        checkbox_grid = QGridLayout()
        for idx, el in enumerate(list_of_elements):
            new_checkbox = QCheckBox(el, self)
            new_checkbox.setChecked(default_choices[idx])
            self.checkboxes.append(new_checkbox)
            current_row = idx//self.MAX_CHECKBOX_COLUMNS
            current_col = idx%self.MAX_CHECKBOX_COLUMNS
            checkbox_grid.addWidget(new_checkbox, current_row, current_col)

        g_layout = QVBoxLayout()
        g_layout.addLayout(checkbox_grid)
        g_layout.addWidget(self.button_box, alignment=Qt.AlignCenter)
        self.setLayout(g_layout)

    def exec_(self, *args, **kwargs):
        """
        Overloading the exec_() function to get both the messagebox result,
        as well as the selected checkboxes.
        """
        ok = QMessageBox.exec_(self, *args, *kwargs)
        accepted_idxs = list()
        for idx, chk in enumerate(self.checkboxes):
            if chk.isChecked():
                accepted_idxs.append(idx)
        return ok, accepted_idxs

class PortSelectionDialog(QDialog):
    """
    Select a port from a list of available system ports.
    """

    def __init__(self):
        QDialog.__init__(self)
        self.setWindowTitle('Select COM Port')

        # First, get a list of available communication ports. For the time
        # being, this has been hard-coded for linux.
        port_list = glob.glob('/dev/tty[A-Za-z]*')
        self.port_selection = QComboBox()
        self.port_selection.addItems(port_list)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        g_layout = QVBoxLayout()
        g_layout.addWidget(self.port_selection)
        g_layout.addWidget(self.button_box, alignment=Qt.AlignCenter)
        self.setLayout(g_layout)

    def exec_(self, *args, **kwargs):
        """
        Overloading the exec_() function to get both the messagebox result,
        as well as the selected entry.
        """
        ok = QMessageBox.exec_(self, *args, *kwargs)
        return ok, self.port_selection.currentText()

class PositionSelectionGUI(QDialog):
    """
    Select a position (coordinate and radius) to designate as a target zone for
    stimulation.
    """

    def __init__(self):
        pass

class FileOpenApplication(QWidget):

    """
    Overkill application to just get an open filename
    """

    def __init__(self):
        """TODO: to be defined1. """
        QWidget.__init__(self)

    def getOpenFileName(self, data_dir, message, file_format):
        """
        Get the file path for a user-selected file.
        """
        data_file, _ = QFileDialog.getOpenFileName(self, message,
                data_dir, file_format)
        return data_file

    def getSaveFileName(self, data_dir, message, file_format):
        save_file, _ = QFileDialog.getSaveFileName(self, message, data_dir,\
                file_format)
        return save_file

    def getDirectory(self, message):
        return QFileDialog.getExistingDirectory(self, message)

# On OSX, getting TK to work seems to be a pain. This is an easier alternative to get the filename
def get_open_file_name(data_dir=None, file_format='All Files (*.*)', message="Choose file"):
    file_dialog = FileOpenApplication()
    if data_dir is None:
        data_dir = os.getcwd()
    file_name = file_dialog.getOpenFileName(str(data_dir), message, file_format)
    file_dialog.close()
    return file_name

def get_save_file_name(data_dir=None, file_format='All Files (*.*)', message="Choose file"):
    file_dialog = FileOpenApplication()
    if data_dir is None:
        data_dir = os.getcwd()
    file_name = file_dialog.getSaveFileName(str(data_dir), message, file_format)
    file_dialog.close()
    return file_name

def get_directory(message="Choose a directory"):
    file_dialog = FileOpenApplication()
    dir_name = file_dialog.getDirectory(message)
    file_dialog.close()
    return dir_name

def parseQtCommandlineArgs(args):
    """
    Parse commandline arguments to get user configuration

    :args: Commandline arguments received from sys.argv
    :returns: Dictionary with aprsed commandline arguments

    """
    parser = argparse.ArgumentParser(description='Spike-Analysis Qt helper.')
    parser.add_argument('--animal', metavar='<animal-name>', help='Animal name')
    parser.add_argument('--date', metavar='YYYYMMDD', help='Experiment date', type=int)
    parser.add_argument('--reference-tetrode', metavar='<LFP Reference>', \
            help='Tetrode index to be used as reference for ripple detection.', type=int)
    parser.add_argument('--baseline-tetrode', metavar='<LFP Baseline>', \
            help='Tetrode index to be used as a baseline for ripple detection.', type=int)
    parser.add_argument('--output-dir', metavar='<output-directory>', help='Output directory where sorted spike data should be stored')
    args = parser.parse_args()
    # print(args)
    return args

if __name__ == "__main__": # This allows running this module by running this script
	import pathlib
	import sys
	this_files_directory = pathlib.Path(__file__).parent.resolve()
	sys.path.insert(0, str(this_files_directory.parent.resolve()) ) # Add parent directory to access other modules

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer
# from PyQt5.QtConcurrent import run

try:
	from PyQt5 import uic
except ImportError:
	import sip
import sys

import numpy as np
import time
from itertools import product
from rich import print

from MPL_Shared.Temperature_Controller import Temperature_Controller
from MPL_Shared.Temperature_Controller_Settings import TemperatureControllerSettingsWindow
from MPL_Shared.SQL_Controller import Commit_XY_Blob_Data_To_SQL, Connect_To_SQL
from MPL_Shared.IV_Measurement_Assistant import IV_Controller
from MPL_Shared.Async_Iterator import Quitting_Early_Exception, Run_Func_Async, Get_Quit_Early_Flag
from MPL_Shared.Saveable_Session import Saveable_Session

from MPL_Shared.Pad_Description_File import Get_Device_Description_File
from MPL_Shared.GUI_Tools import Popup_Error, Popup_Yes_Or_No, resource_path, Measurement_Sweep_Runner
from MPL_Shared.Threaded_Subsystems import Threaded_Subsystems
from DSO_Controller import *
import Hantek_Python_Imgui

import sys
# sys.path.insert(0, 'D:/School/Processing/Lifetime Data')
from Lifetime_Measurement_Plotter import approximate_maximum, get_lifetime_fit, lifetime_shape

__version__ = '1.00'


Ui_MainWindow, QtBaseClass = uic.loadUiType( resource_path("Lifetime_GUI.ui") ) # GUI layout file.


class Lifetime_Measurement_Assistant_App( QtWidgets.QWidget, Ui_MainWindow, Saveable_Session, Threaded_Subsystems ):

	measurementRequested_signal = QtCore.pyqtSignal(float, float, float, float)

	def __init__(self, parent=None, root_window=None):
		QtWidgets.QWidget.__init__(self, parent)
		Ui_MainWindow.__init__(self)
		self.setupUi(self)

		self.tgain_amounts = [100, 1000, 10000, 100000]
		self.tgain_checkboxes = [self.transimpedance100_checkBox, self.transimpedance1000_checkBox, self.transimpedance10000_checkBox, self.transimpedance100000_checkBox]
		Saveable_Session.__init__( self, text_boxes = [(self.user_lineEdit, "user"),(self.descriptionFilePath_lineEdit, "pad_description_path"),(self.sampleName_lineEdit, "sample_name"),
					   (self.startTemp_lineEdit, "start_T"),(self.endTemp_lineEdit, "end_T"), (self.stepTemp_lineEdit, "step_T")],
					   check_boxes = list(zip( self.tgain_checkboxes, self.tgain_amounts )) )

		self.Init_Subsystems()
		self.Connect_Control_Logic()
		self.Start_Subsystems()

		self.Restore_Session( resource_path( "session.ini" ) )

		self.current_data = None
		self.measurement = None

	def closeEvent( self, event ):
		if self.measurement:
			self.quit_early.set()
			self.measurement.wait()
		self.functionGenerator_widget.Set_Output_Off()
		self.graph.close()
		Threaded_Subsystems.closeEvent(self, event)
		QtWidgets.QWidget.closeEvent(self, event)
		self.config_window.close()

	def Init_Subsystems( self ):
		self.sql_type, self.sql_conn = Connect_To_SQL( resource_path( "configuration.ini" ), config_error_popup=Popup_Yes_Or_No )
		self.config_window = TemperatureControllerSettingsWindow()
		self.measurement = None

		self.quit_early = Get_Quit_Early_Flag()
		status_layout = self.connectionsStatusDisplay_widget.layout()
		subsystems = self.Make_Subsystems( self, status_layout,
		                                   IV_Controller( machine_type="Keysight" ),
		                                   Temperature_Controller( resource_path( "configuration.ini" ) ) )
		self.iv_controller, self.temp_controller = subsystems
		# self.Setup_DSO()

		# self.graph.set_labels( title="Lifetime", x_label=r"Time ($\mathrm{\mu s}$)", y_label="Voltage (V)" )
		self.smuManualController_widget.Connect_To_IV_Controller( self.iv_controller )
		self.dso = None

	def Setup_DSO( self ):
		if self.dso != None:
			del self.dso
		self.graph.set_labels( title="Lifetime", x_label=r"Time ($\mathrm{\mu s}$)", y_label="Voltage (V)" )
		# self.dso_x, self.dso, self.dso_parameters = Hantek_Measurer()
		self.dso = Hantek_Python_Imgui.CHard()
		self.dso.Launch_GUI()
		time.sleep( 8 )
		dso_x = self.dso.Get_Timing()
		self.graph.axis2 = self.graph.ax.twinx()

		self.dso_plot, = self.graph.ax.plot( dso_x * 1E6, np.zeros_like( dso_x ), color="red" )
		self.dso_trigger_plot, = self.graph.axis2.plot( dso_x * 1E-6, np.zeros_like( dso_x ) )
		# self.dso_plot, = self.graph.ax.plot( dso_x * 1E6, np.zeros_like( dso_x ), color="red", animated=True )
		# self.dso_trigger_plot, = self.graph.axis2.plot( dso_x * 1E-6, np.zeros_like( dso_x ), animated=True )
		# self.dso_fit_plot, = self.graph.ax.plot( [], [], '.--' )
		# self.graph.ax.relim()
		# self.graph.ax.autoscale_view()
		# self.graph.ax.set_xlim( 0, 1E6 * np.max( dso_x ) )
		# self.graph.ax.set_ylim( -10E-3, 600E-3 )
		# # self.graph.axis2.relim()
		# # self.graph.axis2.autoscale_view()
		# self.graph.axis2.set_xlim( 0, 1E6 * np.max( dso_x ) )
		# self.graph.axis2.set_ylim( -5., 5. )
		# self.graph.figure.tight_layout()

		# self.redraw_func = self.graph.start_live_updates( list_of_graphs=[self.dso_plot, self.dso_trigger_plot] )
									#    replot_function=self.Update_Oscilloscope_Data, update_interval_ms=1000 )
		# QTimer.singleShot(100, self.graph.prepare_blit_box)
		QTimer.singleShot(2000, lambda : self.Update_Oscilloscope_Data( self.dso.Get_Timing(), self.dso.Get_Data( 0 )) )

	def Update_Oscilloscope_Data( self, x, y ):
		# self.graph.new_plot()
		# self.graph.current_graph_data = (x, y)

		x = x * 1E6

		x_offset = approximate_maximum( x, y, 1 )
		shifted_x = x - x_offset
		y = y - np.min( y )
		# (amplitude, lifetime, offset), x_fit_range = get_lifetime_fit( x, y )
		# self.dso_fit_plot.set_xdata( x[x_fit_range] )
		# self.dso_fit_plot.set_ydata( lifetime_shape( x[x_fit_range], amplitude, lifetime, offset ) )
		# self.dso_plot.set_xdata( shifted_x )
		self.dso_plot.set_xdata( x )
		self.dso_plot.set_ydata( y )

		y2 = self.dso.Get_Data( 1 )
		self.dso_trigger_plot.set_xdata( x )
		self.dso_trigger_plot.set_ydata( y2 )

		self.graph.ax.autoscale()
		self.graph.axis2.autoscale()

		# self.graph.ax.autoscale_view(True,True,True)
		self.graph.figure.tight_layout()
		self.graph.canvas.draw()
		self.graph.canvas.flush_events()
		# self.graph.canvas.show()
		# self.redraw_func()

	def Get_Oscilloscope_Data( self ):
		x = self.dso.Get_Timing()
		y = self.dso.Get_Data( 0 )
		QTimer.singleShot(1000, self.Update_Oscilloscope_Data)
		x_offset = approximate_maximum( x, y, 1 )
		shifted_x = x - x_offset
		y = y - np.min( y )
		# (amplitude, lifetime, offset), x_fit_range = get_lifetime_fit( x, y )

		return x, y, None

	def Open_Config_Window( self ):
		self.config_window.show()
		getattr(self.config_window, "raise")() # Bring window to the front even if it's already open
		self.config_window.activateWindow()

	def Connect_Control_Logic( self ):
		self.Stop_Measurement() # Initializes Measurement Sweep Button

		#self.establishComms_pushButton.clicked.connect( self.Establish_Comms )
		self.takeMeasurement_pushButton.clicked.connect( self.Take_Single_Measurement )
		self.outputToFile_pushButton.clicked.connect( self.Save_Data_To_File )
		self.saveToDatabase_pushButton.clicked.connect( self.Save_Data_To_Database )
		self.turnOnDSO_pushButton.clicked.connect( self.Setup_DSO )
		self.turnOnDSO_pushButton.clicked.connect( lambda : self.turnOnDSO_pushButton.setEnabled( False ) )
		self.temperatureHold_widget.Connect_To_Temperature_Controller( self.temp_controller )

		self.measurementRequested_signal.connect( self.iv_controller.Voltage_Sweep )
		# self.iv_controller.newSweepStarted_signal.connect( self.graph.new_plot )
		# self.iv_controller.dataPointGotten_signal.connect( self.graph.add_new_data_point )
		# self.iv_controller.sweepFinished_signal.connect( self.graph.plot_finished )

		# Temperature controller stuff
		self.config_window.Connect_Functions( self.temp_controller )
		self.settings_pushButton.clicked.connect( self.Open_Config_Window )
		self.loadDevicesFile_pushButton.clicked.connect( self.Select_Device_File )

		# Update labels on connection and disconnection to wifi devices
		self.temp_controller.Temperature_Changed.connect( lambda temperature : self.currentTemp_lineEdit.setText( '{:.2f}'.format( temperature ) ) )
		self.temp_controller.PID_Output_Changed.connect( lambda pid_output : self.outputPower_lineEdit.setText( '{:.2f} %'.format( pid_output ) ) )



	def Select_Device_File( self ):
		fileName, _ = QFileDialog.getOpenFileName( self, "QFileDialog.getSaveFileName()", "", "CSV Files (*.csv);;All Files (*)" )
		if fileName == "": # User cancelled
			return
		try:
			config_info = Get_Device_Description_File( fileName )
		except Exception as e:
			Popup_Error( "Error", str(e) )
			return

		self.descriptionFilePath_lineEdit.setText( fileName )

	def Get_Measurement_Sweep_User_Input( self ):
		sample_name = self.sampleName_lineEdit.text()
		user = str( self.user_lineEdit.text() )
		if( sample_name == "" or user == "" ):
			raise ValueError( "Must enter a sample name and user" )

		v_bias = 1E-3 * self.smuManualController_widget.bias_mV_doubleSpinBox.value()
		try:
			temp_start, temp_end, temp_step = float(self.startTemp_lineEdit.text()), float(self.endTemp_lineEdit.text()), float(self.stepTemp_lineEdit.text())
		except ValueError:
			raise ValueError( "Invalid arguement for temperature" )

		device_config_data = Get_Device_Description_File( self.descriptionFilePath_lineEdit.text() )

		transimpedance_values_to_run = [tgain for box, tgain in zip( self.tgain_checkboxes, self.tgain_amounts ) if box.isChecked()]
		if len( transimpedance_values_to_run ) == 0:
			raise ValueError( "Need to select at least one transimpedance gain value" )
		self.sql_type, self.sql_conn = Connect_To_SQL( resource_path( "configuration.ini" ) )
		meta_data = dict( sample_name=sample_name, user=user, laser_wavelength_nm=905 )

		return meta_data, (temp_start, temp_end, temp_step), v_bias, device_config_data, transimpedance_values_to_run

	def Error_During_Measurement( self, error ):
		self.quit_early.set()
		self.Make_Safe()
		Popup_Error( "Error During Measurement:", error )

	def Start_Measurement( self ):
		# Update button to reuse it for stopping measurement
		try:
			self.Save_Session( resource_path( "session.ini" ) )
			self.quit_early.clear()
			self.measurement = Measurement_Sweep_Runner( self, self.Stop_Measurement, self.quit_early, Measurement_Sweep,
			                                             self, self.temp_controller, self.iv_controller,
			                                             *self.Get_Measurement_Sweep_User_Input() )
		except Exception as e:
			Popup_Error( "Error Starting Measurement", str(e) )
			return

		# Update button to reuse it for stopping measurement
		try: self.takeMeasurementSweep_pushButton.clicked.disconnect()
		except Exception: pass
		self.takeMeasurementSweep_pushButton.setText( "Stop Measurement" )
		self.takeMeasurementSweep_pushButton.setStyleSheet("QPushButton { background-color: rgba(255,0,0,255); color: rgba(0, 0, 0,255); }")
		self.takeMeasurementSweep_pushButton.clicked.connect( self.Stop_Measurement )


	def Stop_Measurement( self ):
		self.quit_early.set()

		try: self.takeMeasurementSweep_pushButton.clicked.disconnect()
		except Exception: pass
		self.takeMeasurementSweep_pushButton.setText( "Measurement Sweep" )
		self.takeMeasurementSweep_pushButton.setStyleSheet("QPushButton { background-color: rgba(0,255,0,255); color: rgba(0, 0, 0,255); }")
		self.takeMeasurementSweep_pushButton.clicked.connect( self.Start_Measurement )

	def Set_Current_Data( self, x_data, y_data ):
		self.current_data = ( x_data, y_data )
		self.iv_controller.sweepFinished_signal.disconnect( self.Set_Current_Data )

	def Take_Single_Measurement( self ):
		input_start = float( self.startVoltage_lineEdit.text() )
		input_end = float( self.endVoltage_lineEdit.text() )
		input_step = float( self.stepVoltage_lineEdit.text() )
		time_interval = float( self.timeInterval_lineEdit.text() )
		self.iv_controller.sweepFinished_signal.connect( self.Set_Current_Data )
		self.measurementRequested_signal.emit( input_start, input_end, input_step, time_interval )

	def Save_Data_To_File( self ):
		if self.sampleName_lineEdit.text() == '':
			Popup_Error( "Error", "Must enter sample name" )
			return

		timestr = time.strftime("%Y%m%d-%H%M%S")
		sample_name = str( self.sampleName_lineEdit.text() )

		file_name = "IV Data_" + sample_name + "_" + timestr + ".csv"
		print( "Saving File: " + file_name )
		with open( file_name, 'w' ) as outfile:
			for x,y in zip( self.current_data[0], self.current_data[1] ):
				outfile.write( f'{x},{y}\n' )

	def Save_Data_To_Database( self ):
		if self.current_data == None:
			return

		sample_name = str( self.sampleName_lineEdit.text() )
		user = str( self.user_lineEdit.text() )
		if sample_name == ''  or user == '':
			Popup_Error( "Error", "Must enter sample name and user" )
			return

		meta_data_sql_entries = dict( sample_name=sample_name, user=user, temperature_in_k=None, measurement_setup="Microprobe",
					device_location=None, device_side_length_in_um=None, blackbody_temperature_in_c=None,
					bandpass_filter=None, aperture_radius_in_m=None )

		Commit_XY_Data_To_SQL( self.sql_type, self.sql_conn, xy_data_sql_table="iv_raw_data", xy_sql_labels=("voltage_v","current_a"),
						   x_data=self.current_data[0], y_data=self.current_data[1], metadata_sql_table="iv_measurements", **meta_data_sql_entries )

		print( "Data committed to database: " + sample_name  )

def Get_Good_Data_Run( main_window ):
	num_samples = 16
	# print( "Measurement 0" )
	params = main_window.dso.Get_Settings( 0 )
	dso_parameters = dict( dso_timebase=params.dso_timebase,
						dso_voltagebase=params.voltagebase,
						dso_coupling=params.coupling,
						dso_buffer_length=params.buffer_length )

	x_data, total_y_data = main_window.dso.Get_Timing(), main_window.dso.Get_Data( 0 )
	for _ in range(num_samples - 1):
		# print( f"Measurement {_+1}" )
		x_data, y_data = main_window.dso.Get_Timing(), main_window.dso.Get_Data( 0 )
		total_y_data += y_data
	total_y_data /= num_samples

	return x_data, y_data, None, dso_parameters


def Measurement_Sweep( quit_early,
                       main_window, temp_controller, iv_controller,
                       meta_data, temperature_info, bias_v, device_config_data, transimpedance_values_to_run ):
	Run_Func_Async( temp_controller, temp_controller.Make_Safe )
	Run_Func_Async( iv_controller, iv_controller.Make_Safe )
	sql_type, sql_conn = Connect_To_SQL( resource_path( "configuration.ini" ) )

	# if temperature_info is None:
	temp_start, temp_end, temp_step = temperature_info
	if temp_step == 0:
		temp_range = np.concatenate( [np.arange( 80, 101, 5 ), np.arange( 110, 300, 10 )] )
	else:
		temp_range = np.arange( temp_start, temp_end + temp_step / 2, temp_step )
	original_params = main_window.dso.Get_Settings( 0 )

	try:
		for set_temp, device, transimpedance_gain in product(
			temp_range,
			device_config_data, transimpedance_values_to_run ):
			Run_Func_Async( temp_controller, temp_controller.Set_Transimpedance_Gain, transimpedance_gain )
			Run_Func_Async( temp_controller, temp_controller.Set_Temp_And_Turn_On, set_temp )
			actual_temperature = Run_Func_Async( temp_controller, temp_controller.Wait_For_Stable_Temperature )

			# "neg_pad", "pos_pad", "side", "location"
			(neg_pad, pos_pad), pads_are_reversed = Run_Func_Async( temp_controller, temp_controller.Set_Active_Pads, device.neg_pad, device.pos_pad )
			pads_are_reversed = pads_are_reversed or (neg_pad == 8 and pos_pad == 15)
			used_bias_v = -bias_v if pads_are_reversed else bias_v

			current_a = Run_Func_Async( iv_controller, iv_controller.Set_Bias, used_bias_v )
			Run_Func_Async( temp_controller, temp_controller.Turn_Off )
			Run_Func_Async( main_window, main_window.functionGenerator_widget.Set_Output_On )
			time.sleep( 1 )
			lever_position = 255 - original_params.lever_position if pads_are_reversed else original_params.lever_position
			main_window.dso.Set_Lever_Pos( 0, int(lever_position) )

			meta_data.update( dict( bias_v=bias_v, transimpedance_gain=transimpedance_gain,
					temperature_in_k=actual_temperature, device_location=device.location, device_side_length_um=device.side ) )
			print( f"Starting Measurement for {device.location} side length {device.side} at {actual_temperature} K on pads {neg_pad} and {pos_pad}" )

			x_data, y_data, lifetime_guess_s, dso_parameters = Get_Good_Data_Run( main_window )
			y_data = -y_data if pads_are_reversed else y_data
			Run_Func_Async( main_window, lambda : main_window.Update_Oscilloscope_Data(x_data, y_data) )
			meta_data["lifetime_guess_s"] = lifetime_guess_s

			Run_Func_Async( main_window, main_window.functionGenerator_widget.Set_Output_Off )
			Run_Func_Async( iv_controller, iv_controller.Turn_Off_Bias )
			meta_data.update( dso_parameters )
			Commit_XY_Blob_Data_To_SQL( sql_type, sql_conn, xy_data_sql_table="lifetime_raw_data", xy_sql_labels=("time_s","voltage_v"),
								x_data=x_data, y_data=y_data, metadata_sql_table="lifetime_measurements", **meta_data )
	except Quitting_Early_Exception as e:
		print( "Quitting Early" )
		quit_early.clear()
	except Exception as e:
		print( "Measurement loop ended in exception:", e )

	print( "Quitting Early0" )
	Run_Func_Async( main_window, main_window.functionGenerator_widget.Set_Output_Off )
	print( "Quitting Early1" )
	Run_Func_Async( iv_controller, iv_controller.Turn_Off_Bias )
	print( "Quitting Early2" )
	Run_Func_Async( temp_controller, temp_controller.Make_Safe )
	print( "Quitting Early3" )
	Run_Func_Async( iv_controller, iv_controller.Make_Safe )

	print( "Finished Measurment" )


if __name__ == "__main__":
	app = QtWidgets.QApplication( sys.argv )
	window = Lifetime_Measurement_Assistant_App()
	window.show()
	sys.exit( app.exec_() )

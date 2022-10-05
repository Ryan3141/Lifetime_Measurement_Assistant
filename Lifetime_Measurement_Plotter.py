from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from rich import print
from matplotlib.pyplot import figure

def load_oscilloscope_file( file_path ):
	with open( file_path, 'r' ) as file:
		file_contents = file.read()
	si_unit_conversion = { 'p':1E-12, 'n':1E-9, 'u':1E-6, 'm':1E-3, ' ':1, 'k':1E3, 'M':1E6, 'G':1E9, 'T':1E12 }
	float_regex = r'[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?'
	p = re.compile( r'#CHANNEL:CH(?P<channel>\d)\n'
				   f'#CLOCK=(?P<clock>{float_regex})' r'(?P<time_units>[pnum]?S)\n'
				   r'#SIZE=(?P<size>\d+)\n'
				   r'#UNITS:(?P<y_units>[pnumk]?V)\n\n'
				   f'(?P<data>(?:{float_regex}\\n+)+)' )
#     print( file_contents )
	channels = [ m.groupdict() for m in p.finditer( file_contents ) ]
	channels = { int(channel["channel"]) - 1: channel for channel in channels }
	for channel in channels.values():
		for its_type, number_key in zip( [int, float, int], ["channel", "clock", "size"] ):
			channel[ number_key ] = its_type( channel[ number_key ] )
		channel["time_units"] = channel["time_units"] if len( channel["time_units"] ) == 2 else (" " + channel["time_units"])
		channel["y_units"] = channel["y_units"] if len( channel["y_units"] ) == 2 else (" " + channel["y_units"])
		channel["data"] = np.array( [float(x) for x in channel["data"].split('\n') if x != ""] )
		channel["signal_V"] = np.array( channel["data"] ) * si_unit_conversion[ channel["y_units"][0] ]
		# channel["signal_V"] = np.array( channel["data"][:channel["size"] // 4] ) * si_unit_conversion[ channel["y_units"][0] ]
		channel["time_S"] = np.linspace( 0, 1E-3 * channel["clock"] * channel["size"] * 4, len( channel["signal_V"] ), endpoint=False ) * si_unit_conversion[ channel["time_units"][0] ]
	return channels

def approximate_maximum( x, y, number_of_samples = 60 ):
	top_values = sorted( zip( x, y ), key=lambda one_pair : one_pair[1] )[-number_of_samples:]
	# if number_of_samples > 1:
	# 	domain_top = np.max( top_values ) - np.min( top_values )
	# 	domain = np.max( x ) - np.min( x )
	# 	if domain_top / domain > 0.4: # If the range of max values is too big, averaging is a bad idea
	# 		return top_values[0][0] # Just return the highest value

	average_max_x = np.average( np.array([data[0] for data in top_values]) )
	return average_max_x

def lifetime_shape( t, amplitude, tau, offset ):
	return amplitude * np.exp( -(t - offset) / tau )

def find_first_zero_crossing( x_data, y_data ):
	y_was_positive = y_data[0] > 0
	for x, y in zip( x_data, y_data ):
		if y <= 0 and y_was_positive:
			return x
		elif y > 0 and not y_was_positive:
			return x

def derivative( x, y ):
	x_avg = (x[1:] + x[:-1]) / 2
	dx = x[1:] - x[:-1]
	dy = y[1:] - y[:-1]
	return x_avg, (dy / dx)

def get_lifetime_fit( x, y ):
	resample_x = np.array( [ np.average( x[n:n+8] ) for n in range( 0, len(x), 8 ) ] )
	resample_y = np.array( [ np.average( y[n:n+8] ) for n in range( 0, len(y), 8 ) ] )
	dy = resample_y[:-1] - resample_y[1:]
	dx = resample_x[:-1] - resample_x[1:]

	try:
		end_of_lifetime = find_first_zero_crossing( resample_x[resample_x > 0.1][:-1], dy[resample_x[:-1] > 0.1] )
		x_fit_range = (x >= 0.15 * end_of_lifetime) & (x < 0.5 * end_of_lifetime)
		(amplitude, lifetime, offset), _ = curve_fit( lifetime_shape, x[x_fit_range], y[x_fit_range], bounds=([0, 0, -20],[np.inf, 100, +20]) )
		# print( f"Lifetime = {lifetime:0.3}us, Amplitude = {amplitude:0.3}, offset = {offset:0.3} us" )
		# print( f"Lifetime = {lifetime:0.3}us, Amplitude = {amplitude:0.3}, offset = {start_offset:0.3}")
		# axis.plot( x[x_fit_range], lifetime_shape( x[x_fit_range], amplitude, lifetime, offset ), '.--', label=f"Fit $\\tau={lifetime:0.3}$" r" $\mathrm{\mu s}$", color=color )
		# axis.plot( x[x_fit_range], lifetime_shape( x[x_fit_range], amplitude, lifetime, offset, y_offset ) / np.max(y), '.--', label=f"Fit $\\tau={lifetime:0.3}$" r" $\mathrm{\mu s}$", color=color )
		# axis.plot( x[x_fit_range], np.log10( np.abs(y[x_fit_range]) + 1E-6 ), label=f"{Path(file_name).stem}", color=color )
		return (amplitude, lifetime, offset), x_fit_range
	except Exception as e:
		return (1, 1E-18, 0), np.full_like( x, fill_value=True, dtype=bool )
		pass

def plot_raw_lifetime_data( lifetime_datasets, channel_with_lifetime_data, all_files_to_use ):
	fig, axis = plt.subplots( 1, 1, figsize=(12,8) )
	axis2 = axis.twinx()
	for dataset, file_name, color in zip( lifetime_datasets, all_files_to_use, cm.rainbow(np.linspace(1, 0, len(lifetime_datasets))) ):
		channel = dataset[channel_with_lifetime_data]
		print( ' '.join( f"{x}: {channel[x]}" for x in channel.keys() if x not in [ "data", "signal_V", "time_S" ] ) )
		y = channel["signal_V"]
		x = channel["time_S"] * 1E6
		x_offset = approximate_maximum( x, y, 1 )
		x = x - x_offset
		y = y - np.min( y )
		# axis.plot( range(len(dataset[0]["data"])), dataset[0]["data"] * 0.25 / np.max(dataset[0]["data"]), label=f"{Path(file_name).stem}", color=color )
		# axis.plot( range(len(dataset[1]["data"])), dataset[1]["data"] * 0.25 / np.max(dataset[1]["data"]), label=f"{Path(file_name).stem}", color=color )
		# useful_x = (x >= -0.2 * end_of_lifetime) & (x < 1.5 * end_of_lifetime)
		(amplitude, lifetime, offset), x_fit_range = get_lifetime_fit( x, y )
		axis.plot( x[x_fit_range], lifetime_shape( x[x_fit_range], amplitude, lifetime, offset ), '.--', label=f"Fit $\\tau={lifetime:0.3}$" r" $\mathrm{\mu s}$", color=color )
		# axis.plot( x[useful_x], y[useful_x], label=f"{dataset['temperature']} K", color=color )
		temperature_label = Path(file_name).parent.stem
		axis.plot( x, y, label=temperature_label, color=color )
		# axis.plot( x, y, label=f"{Path(file_name).stem}", color=color )
		# axis2.plot( *derivative( resample_x, resample_y ), label=f"{Path(file_name).stem}", color=color )

	axis.set_title( f"{Path(file_name).stem}" )
	axis.set_xlim( -2, 20 )
	# axis.set_xlim( 0, dataset["clock"] )
	axis.set_xlabel(r'time ($\mu s$)')
	axis.set_ylabel(r'Voltage (V)')
	axis.grid()
	axis.legend()
	plt.show( block=True )

def plot_lifetime_vs_temperature( lifetime_datasets, channel_with_lifetime_data, temperatures ):
	# fig, axis = plt.subplots( 1, 1, figsize=(12,8) )
	fig = figure(num=None, figsize=(8, 6), facecolor='w', edgecolor='k')
	axis = fig.subplots()
	x_ax2 = axis.twiny()

	wanted_channel = [ dataset[channel_with_lifetime_data] for dataset in datasets ]
	xy_data_per = [ (channel["time_S"] * 1E6, channel["signal_V"]) for channel in wanted_channel ]
	lifetime_fit_data = [ get_lifetime_fit( x - approximate_maximum( x, y, 1 ), y - np.min( y ) ) for x, y in xy_data_per ]
	lifetimes = [ lifetime for (amplitude, lifetime, offset), x_fit_range in lifetime_fit_data ]

	axis.semilogy( 1000 / np.array(temperatures), lifetimes, '.-' )
	# axis.plot( x[useful_x], y[useful_x], label=f"{dataset['temperature']} K", color=color )
	# axis.plot( x, y, label=f"{Path(file_name).parent}", color=color )
	# axis.plot( x, y, label=f"{Path(file_name).stem}", color=color )
	# axis2.plot( *derivative( resample_x, resample_y ), label=f"{Path(file_name).stem}", color=color )

	# axis.set_title( f"{Path(file_name).stem}" )
	# axis.set_xlim( -2, 20 )
	# axis.set_xlim( 0, dataset["clock"] )

	axis.set_xlim( 1000 / 333, 13 )
	ybottom, ytop = axis.get_ylim()
	# ax.set_ylim( min( 1E-7, ybottom ), max( 1E-4, ytop ) )
	axis.set_ylim( min( 1, ybottom ), max( 1E2, ytop ) )

	axis.set_xlabel(r"1000 / Temperature ($K^{-1}$)")
	axis.set_ylabel(r'Lifetime ($\mathrm{\mu s}$)')
	axis.grid()
	axis.legend()

	new_tick_locations = 1000 / np.array([80, 100, 150, 200, 250, 300])
	def tick_function( X ):
		V = 1000 / X
		return ["%.0f" % z for z in V]
	x_ax2.set_xlim( axis.get_xlim() )
	x_ax2.set_xticks( new_tick_locations )
	x_ax2.set_xticklabels( tick_function( new_tick_locations ) )
	x_ax2.set_xlabel(r"Temperature (K)")

	plt.show( block=True )

def plot_raw_lifetime_data2( title, lifetime_datasets, labels ):
	fig, axis = plt.subplots( 1, 1, figsize=(12,8) )
	axis2 = axis.twinx()
	for dataset, label, color in zip( lifetime_datasets, labels, cm.rainbow(np.linspace(1, 0, len(lifetime_datasets))) ):
		x, y = dataset[1] * 1E6, dataset[2]
		x_offset = approximate_maximum( x, y, 1 )
		x = x - x_offset
		y = y - np.min( y )
		# axis.plot( range(len(dataset[0]["data"])), dataset[0]["data"] * 0.25 / np.max(dataset[0]["data"]), label=f"{Path(file_name).stem}", color=color )
		# axis.plot( range(len(dataset[1]["data"])), dataset[1]["data"] * 0.25 / np.max(dataset[1]["data"]), label=f"{Path(file_name).stem}", color=color )
		# useful_x = (x >= -0.2 * end_of_lifetime) & (x < 1.5 * end_of_lifetime)
		(amplitude, lifetime, offset), x_fit_range = get_lifetime_fit( x, y )
		axis.plot( x[x_fit_range], lifetime_shape( x[x_fit_range], amplitude, lifetime, offset ), '.--', label=f"Fit $\\tau={lifetime:0.3}$" r" $\mathrm{\mu s}$", color=color )
		# axis.plot( x[useful_x], y[useful_x], label=f"{dataset['temperature']} K", color=color )
		resample_x = np.array( [ np.average( x[n:n+32] ) for n in range( 0, len(x), 32 ) ] )
		resample_y = np.array( [ np.average( y[n:n+32] ) for n in range( 0, len(y), 32 ) ] )

		axis.plot( resample_x, resample_y, label=label, color=color )
		# axis.plot( x, y, label=f"{Path(file_name).stem}", color=color )
		# axis2.plot( *derivative( resample_x, resample_y ), label=f"{Path(file_name).stem}", color=color )

	axis.set_title( title )
	# axis.set_xlim( -2, 20 )
	# axis.set_xlim( 0, dataset["clock"] )
	axis.set_xlabel(r'time ($\mu s$)')
	axis.set_ylabel(r'Voltage (V)')
	axis.grid()
	axis.legend()
	plt.show( block=True )

def plot_lifetime_vs_temperature2( title, lifetime_datasets, temperatures ):
	# fig, axis = plt.subplots( 1, 1, figsize=(12,8) )
	fig = figure(num=None, figsize=(8, 6), facecolor='w', edgecolor='k')
	axis = fig.subplots()
	x_ax2 = axis.twiny()

	xy_data_per = [ (dataset[1] * 1E6, dataset[2]) for dataset in lifetime_datasets ]
	lifetime_fit_data = [ get_lifetime_fit( x - approximate_maximum( x, y, 1 ), y - np.min( y ) ) for x, y in xy_data_per ]
	lifetimes = [ lifetime for (amplitude, lifetime, offset), x_fit_range in lifetime_fit_data ]

	axis.semilogy( 1000 / np.array(temperatures), lifetimes, '.-' )
	# axis.plot( x[useful_x], y[useful_x], label=f"{dataset['temperature']} K", color=color )
	# axis.plot( x, y, label=f"{Path(file_name).parent}", color=color )
	# axis.plot( x, y, label=f"{Path(file_name).stem}", color=color )
	# axis2.plot( *derivative( resample_x, resample_y ), label=f"{Path(file_name).stem}", color=color )

	# axis.set_title( f"{Path(file_name).stem}" )
	# axis.set_xlim( -2, 20 )
	# axis.set_xlim( 0, dataset["clock"] )

	axis.set_xlim( 1000 / 333, 13 )
	ybottom, ytop = axis.get_ylim()
	# ax.set_ylim( min( 1E-7, ybottom ), max( 1E-4, ytop ) )
	axis.set_ylim( min( 1, ybottom ), max( 1E2, ytop ) )

	axis.set_xlabel(r"1000 / Temperature ($K^{-1}$)")
	axis.set_ylabel(r'Lifetime ($\mathrm{\mu s}$)')
	axis.grid()
	axis.legend()

	new_tick_locations = 1000 / np.array([80, 100, 150, 200, 250, 300])
	def tick_function( X ):
		V = 1000 / X
		return ["%.0f" % z for z in V]
	x_ax2.set_xlim( axis.get_xlim() )
	x_ax2.set_xticks( new_tick_locations )
	x_ax2.set_xticklabels( tick_function( new_tick_locations ) )
	x_ax2.set_xlabel(r"Temperature (K)")

	plt.show( block=True )

def Test_MySQL():
	import pathlib
	import sys
	this_files_directory = pathlib.Path(__file__).parent.resolve()
	sys.path.insert(0, str(this_files_directory.parent.resolve()) ) # Add parent directory to access other modules
	from MPL_Shared.GUI_Tools import resource_path
	from MPL_Shared.SQL_Controller import Read_XY_Blob_Data_From_SQL, Connect_To_SQL

	nice_name = dict( sample_name="Sample", device_side_length_um="Size", device_location="Location", bias_v="Bias", transimpedance_gain="Gain", temperature_in_k="T" )
	# filter_by_this = [ ("measurement_id","=","1001") ]
	filter_by_this = [ ("sample_name","=","'R22-C'"), ("transimpedance_gain","=",1E5), ("device_side_length_um","=",200), ("device_location","=","'Single Mesa Left Side'"), ("bias_v","=",-0.3) ]#, ("transimpedance_gain","=",1E5) ] #
	# filter_by_this = [ ("sample_name","=","'R19-E'"), ("transimpedance_gain","=",1E4), ("device_side_length_um","=",200), ("device_location","=","'Single Mesa Right Side'"), ("bias_v","=",-0.3) ]#, ("transimpedance_gain","=",1E5) ] #
	# filter_by_this = [ ("sample_name","=","'R23-A'"), ("transimpedance_gain","=",1E5) ]#, ("transimpedance_gain","=",1E5) ] #
	sql_type, sql_conn = Connect_To_SQL( resource_path( "configuration.ini" ) )
	metadata_labels = ["temperature_in_k"]
	# metadata_labels = ["device_location", "device_side_length_um","temperature_in_k"]
	# metadata_labels = ["transimpedance_gain", "bias_v", "temperature_in_k"]
	metadata, xy_data = Read_XY_Blob_Data_From_SQL( sql_type, sql_conn, filter=" AND ".join( f"{key}{comparator}{value}" for key, comparator, value in filter_by_this ),
													xy_data_sql_table="lifetime_raw_data", xy_sql_labels=("time_s","voltage_v"),
													metadata_sql_table="lifetime_measurements", metadata_requested=metadata_labels )

	title = " AND ".join( f"{nice_name[key]}={value}" for key, comparator, value in filter_by_this )
	labels = [ " ".join( f"{nice_name[key]}={value}" for key, value in zip( metadata_labels, one_metadata[1:] ) ) for one_metadata in metadata ]
	plot_raw_lifetime_data2( title, xy_data, labels )
	temperatures = [ x[metadata_labels.index("temperature_in_k") + 1] for x in metadata ]
	plot_lifetime_vs_temperature2( title, xy_data, temperatures )

def Quick_Fix():
	import pathlib
	import sys
	this_files_directory = pathlib.Path(__file__).parent.resolve()
	sys.path.insert(0, str(this_files_directory.parent.resolve()) ) # Add parent directory to access other modules
	from MPL_Shared.GUI_Tools import resource_path
	from MPL_Shared.SQL_Controller import Read_XY_Blob_Data_From_SQL, Connect_To_SQL

	nice_name = dict( sample_name="Sample", device_side_length_um="Size", device_location="Location", bias_v="Bias", transimpedance_gain="Gain", temperature_in_k="T" )
	filter_by_this = [ ("sample_name","=","'R19-E'"), ("transimpedance_gain","=",1E4), ("device_side_length_um","=",200), ("device_location","=","'Single Mesa Right Side'"), ("bias_v","=",-0.3) ]#, ("transimpedance_gain","=",1E5) ] #
	sql_type, sql_conn = Connect_To_SQL( resource_path( "configuration.ini" ) )
	metadata_labels = ["device_location", "device_side_length_um","temperature_in_k"]
	# metadata_labels = ["transimpedance_gain", "bias_v", "temperature_in_k"]
	metadata, xy_data = Read_XY_Blob_Data_From_SQL( sql_type, sql_conn, filter=" AND ".join( f"{key}{comparator}{value}" for key, comparator, value in filter_by_this ),
													xy_data_sql_table="lifetime_raw_data", xy_sql_labels=("time_s","voltage_v"),
													metadata_sql_table="lifetime_measurements", metadata_requested=metadata_labels )

	title = " AND ".join( f"{nice_name[key]}={value}" for key, comparator, value in filter_by_this )
	labels = [ " ".join( f"{nice_name[key]}={value}" for key, value in zip( metadata_labels, one_metadata[1:] ) ) for one_metadata in metadata ]
	plot_raw_lifetime_data2( title, xy_data, labels )
	temperatures = [ x[metadata_labels.index("temperature_in_k") + 1] for x in metadata ]


if __name__ == "__main__":
	Test_MySQL()
	exit()
	# temperatures = [80, 85, 90, 95, 100, 110, 122, 132, 142, 151, 160, 174, 180]
	# temperatures = [80, 85, 90, 95, 100, 110, 122, 132, 142, 151, 160, 174, 180, 190]
	# all_files_to_use = [ "Reference 3 kHz Wave 20us 5V.txt", "Reference 5 kHz Wave 20us 5V.txt", "Reference 4 kHz Wave 50us 5V.txt", "Reference 5 kHz Wave 20us 5V 16k sampling.txt", "Reference 5 kHz Wave 20us 1V.txt", "Reference 5 kHz Wave 20us 5V 64k sampling.txt" ]
	# all_files_to_use = [ "R19-B 84.1 K.txt", "R19-B 100 K.txt", "R19-B 90 K.txt", "R19-B 250 K.txt", "R19-B 238 K.txt", "R19-B 220 K.txt", "R19-B 211 K.txt" ]
	# all_files_to_use = [ r"R23-A\R23-A_300um_Right_Side 105 K.txt" ]
	# all_files_to_use = [ f"{T} K\\R20-D 100um Right Side -300 mV.txt" for T in [190] ]

	temperatures = [80, 85, 90, 95, 100, 110, 122, 132, 142, 151, 160, 174, 180]
	all_files_to_use = [ f"R20-D\\{T} K\\R20-D 100um Right Side -300 mV.txt" for T in temperatures ]
	# temperatures = [80, 90, 95, 100, 110, 122, 132, 142, 151, 160, 174, 180, 190]
	# all_files_to_use = [ f"{T} K\\R20-D 150um Right Side -300 mV.txt" for T in temperatures ]

	# all_files_to_use = ["R19-B 197 K.txt", "R19-B 270 K.txt"]
	all_files_to_use = sorted( all_files_to_use, key=lambda f : float( f.split(' ')[-2] ) )


	# file_location = Path( r"C:\Code\MPL_Toolbox_From_Repo_Fresh\Lifetime_Measurement_Assistant\Data" )
	file_location = Path( r"D:\School\Processing\Lifetime Data" )
	# file_name = Path( r"Reference 1 kHz Wave.txt" )
	# channels = load_oscilloscope_file( file_location / file_name )
	channel_with_lifetime_data = 0
	datasets = [ load_oscilloscope_file( file_location / file_name ) for file_name in all_files_to_use ]
	# for dataset, file_name in zip( datasets, all_files_to_use ):
	# 	try:
	# 		dataset["temperature"] = float( file_name.split(' ')[1] )
	# 	except Exception as e:
	# 		pass
		# file_name = Path( r"Test Lifetime R19-B 177.5 K.txt" )
	plot_raw_lifetime_data( datasets, channel_with_lifetime_data, all_files_to_use )
	plot_lifetime_vs_temperature( datasets, channel_with_lifetime_data, temperatures )
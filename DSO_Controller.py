import numpy as np
import time
import Hantek_Python_Imgui
# print( Hantek_Python.greet() )

class Coupling:
	DC = 0
	AC = 1
	GND = 2

coupling_string = [
	"DC",
	"AC",
	"GND" ]

class Buffer:
	S4K  = 0x1000
	S8K  = 0x2000
	S16K = 0x4000
	S32K = 0x8000
	S64K = 0x10000

class TriggerSlope:
	RISE = 0
	FALL = 1

class TriggerSweep:
	AUTO = 0
	NORMAL = 1
	SINGLE = 2

class TriggerMode:
	EDGE  = 0
	PULSE = 1
	VIDEO = 2
	CAN   = 3
	LIN   = 4
	UART  = 5
	SPI   = 6
	IIC   = 7

Index_To_Time_Base = [ 2E-9, 5E-9, 10E-9, 20E-9, 50E-9, 100E-9, 200E-9, 500E-9, # In seconds
				 1E-6, 2E-6, 5E-6, 10E-6, 20E-6, 50E-6, 100E-6, 200E-6, 500E-6,
				 1E-3, 2E-3, 5E-3, 10E-3, 20E-3, 50E-3, 100E-3, 200E-3, 500E-3,
				 1E+0, 2E+0, 5E+0, 10E+0, 20E+0, 50E+0, 100E+0, 200E+0, 500E+0,
				 1E3 ]

Index_To_Voltage_Base = [ 2E-3, 5E-3, 10E-3, 20E-3, 50E-3, 100E-3, 200E-3, 500E-3, # In Volts
					  1., 2.  , 5.  , 10. ]


# class debug:
# 	def __init__( self ):
# 		rng = np.random.default_rng()
# 	def ReadData( self, _ ):
# 		return np.zeros( 2**16 )
# 		# return np.random.rand(2**16)
# def Hantek_Measurer():
# 	try:
# 		dso = Hantek_Python.CHard()
# 	except Exception as e:
# 		print( "Issue starting Hantek" )
# 		print( e )
# 	# dso = Hantek_Python.CHard()
# 	if not dso.FindeDev():
# 		return np.arange( 2**16 ), debug(), {}
# 		return None, None, None

# 	dso_timebase, dso_voltagebase, dso_coupling, dso_buffer_length = 2E-6, 50E-3, Coupling.AC, Buffer.S32K
# 	for index, (turn_on, coupling, voltagebase, y_lever) in enumerate( zip([True, True, False, False],
# 															      [dso_coupling, Coupling.DC, Coupling.DC, Coupling.DC],
# 															      [dso_voltagebase, 500E-3, 1., 1.],
# 															      [100, 0, 0, 0]) ):
# 		dso.Setup_Channel( index, turn_on, Index_To_Voltage_Base.index( voltagebase ), coupling, y_lever )
# 	time = dso.Setup_Timing( Index_To_Time_Base.index( dso_timebase ), dso_buffer_length )
# 	dso.Setup_Trigger( 1, 125, TriggerSlope.RISE, TriggerSweep.AUTO, TriggerMode.EDGE )
# 	dso_parameters = dict( dso_timebase=dso_timebase, dso_voltagebase=dso_voltagebase, dso_coupling=coupling_string[dso_coupling], dso_buffer_length=dso_buffer_length )
# 	dso.Lock_In_Changes()
# 	return time, dso, dso_parameters
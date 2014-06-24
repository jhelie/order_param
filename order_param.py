################################################################################################################################################
# IMPORT MODULES
################################################################################################################################################

#import general python tools
import argparse
import operator
from operator import itemgetter
import sys, os, shutil
import os.path
import math

#import python extensions/packages to manipulate arrays
import numpy 				#to manipulate arrays
import scipy 				#mathematical tools and recipesimport MDAnalysis

#import graph building module
import matplotlib as mpl
mpl.use('Agg')
import pylab as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm				#colours library
import matplotlib.ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
fontP=FontProperties()

#import MDAnalysis
import MDAnalysis
from MDAnalysis import *
import MDAnalysis.analysis
import MDAnalysis.analysis.leaflet
import MDAnalysis.analysis.distances

#set MDAnalysis to use periodic boundary conditions
MDAnalysis.core.flags['use_periodic_selections'] = True
MDAnalysis.core.flags['use_KDTree_routines'] = False
MDAnalysis.core.flags['use_KDTree_routines'] = False

################################################################################################################################################
# RETRIEVE USER INPUTS
################################################################################################################################################

#create parser
#=============
version_nb="0.1.2"
parser = argparse.ArgumentParser(prog='order_param', usage='', add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter, description=\
'''
**********************************************
v''' + version_nb + '''
author: Jean Helie (jean.helie@bioch.ox.ac.uk)
git: https://github.com/jhelie/order_param
**********************************************

[ Description ]

This script computes the second rank order parameter as defined by:

P2 = 0.5*(3*<cos**2(theta)> - 1)

where theta is the angle between the bond and the bilayer normal.
 P2 = 1      perfect alignement with the bilayer normal
 P2 = 0      random orientation
 P2 = -0.5   anti-alignement

The script produces the following outputs:
 - (time evolution of) P2 for each lipid specie in each leaflet 
 - (time evolution of) P2 for each flipflopping lipid (if any, see note 3) 	

[ Requirements ]

The following python module(s) are needed:
 - MDAnalysis

[ Notes ]

1. It's a good idea to pre-process the xtc first:
    - use trjconv with the -pbc mol option
    - only output the relevant lipids (e.g. no water but no cholesterol either)

2. The z axis is considered to be the bilayer normal. The more your system deforms, the further from
   the actual bilayer normal the z axis will be.

3. In case lipids flipflop during the trajectory, a file listing them can be supplied via the -i flag.
   This file can be the output of the ff_detect script and should follow the format:
   'resname,resid,starting_leaflet' format on each line e.g. 'POPC,145,lower'
   If flipflopping lipids are not identified they may add some significant noise to the results

4. The code can easily be updated to add more lipids, for now the following tails can be dealt with:
    - Martini: DHPC,DHPE,DLPC,DLPE,DAPC,DUPC,DPPC,DPPE,DPPS,DPPG,DSPC,DSPE,POPC,POPE,POPS,POPG,PPCS,PIP2,PIP3,GM3

5. The order parameter calculated for each (handled) lipd specie can be visualised with VMD.
   This can be done either with pdb files (output frequency controled via -w flag) or with the 
   xtc trajectory.
     - pdb file: the order parameter info is stored in the beta factor column. Just open
                 the pdb with VMD and choose Draw Style > Coloring Method > Beta 
     - xtc file: the order parameter info is stored in a .txt file in /3_VMD/ and you can load it into
                 the user field in the xtc by sourcing the script 'set_user_fields.tcl' and running the
                 procedure 'set_order_param'

6. The colour associated to each lipid specie can be defined by supplying a colour file containing
   'resname,colour' on each line (a line with a colour MUST be defined for all species).
   Colours can be specified using single letter code (e.g. 'r'), hex code  or the name of colormap.
   In case a colormap is used, its name must be specified as the colour for each lipid specie - type
   'order_param --colour_maps' to see a list of the standard colour maps.
   If no colour is used the 'jet' colour map is used by default.

[ Usage ]
	
Option	      Default  	Description                    
-----------------------------------------------------
-f			: structure file [.gro]
-x			: trajectory file [.xtc] (optional)
-c			: colour definition file, see note 6
-o			: name of output folder
-b			: beginning time (ns) (the bilayer must exist by then!)
-e			: ending time (ns)	
-t 		10	: process every t-frames
-w			: write annotated pdbs every [w] processed frames (optional, see note 5)
--smooth		: nb of points to use for data smoothing (optional)

Lipids identification  
-----------------------------------------------------
--flipflops		: input file with flipflopping lipids, see note 3
--forcefield		: forcefield options, see note 3
--no-opt		: do not attempt to optimise leaflet identification (useful for huge system)

Other options
-----------------------------------------------------
--colour_maps		: show list of standard colour maps, see note 6
--version		: show version number and exit
-h, --help		: show this menu and exit
  
''')

#data options
parser.add_argument('-f', nargs=1, dest='grofilename', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-x', nargs=1, dest='xtcfilename', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-c', nargs=1, dest='colour_file', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-o', nargs=1, dest='output_folder', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-b', nargs=1, dest='t_start', default=[-1], type=int, help=argparse.SUPPRESS)
parser.add_argument('-e', nargs=1, dest='t_end', default=[10000000000000], type=int, help=argparse.SUPPRESS)
parser.add_argument('-t', nargs=1, dest='frames_dt', default=[10], type=int, help=argparse.SUPPRESS)
parser.add_argument('-w', nargs=1, dest='frames_write_dt', default=[1000000000000000], type=int, help=argparse.SUPPRESS)
parser.add_argument('--smooth', nargs=1, dest='nb_smoothing', default=[0], type=int, help=argparse.SUPPRESS)

#lipids identification
parser.add_argument('--flipflops', nargs=1, dest='selection_file_ff', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('--forcefield', dest='forcefield_opt', choices=['martini'], default='martini', help=argparse.SUPPRESS)
parser.add_argument('--no-opt', dest='cutoff_leaflet', action='store_false', help=argparse.SUPPRESS)

#other options
parser.add_argument('--colour_maps', dest='show_colour_map', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--version', action='version', version='%(prog)s v' + version_nb, help=argparse.SUPPRESS)
parser.add_argument('-h','--help', action='help', help=argparse.SUPPRESS)

#store inputs
#============
args=parser.parse_args()
args.grofilename=args.grofilename[0]
args.xtcfilename=args.xtcfilename[0]
args.colour_file=args.colour_file[0]
args.output_folder=args.output_folder[0]
args.frames_dt=args.frames_dt[0]
args.frames_write_dt=args.frames_write_dt[0]
args.t_start=args.t_start[0]
args.t_end=args.t_end[0]
args.selection_file_ff=args.selection_file_ff[0]
args.nb_smoothing=args.nb_smoothing[0]

#show colour maps
#----------------
if args.show_colour_map:
	print ""
	print "The following standard matplotlib color maps can be used:"
	print ""
	print "Spectral, summer, coolwarm, pink_r, Set1, Set2, Set3, brg_r, Dark2, hot, PuOr_r, afmhot_r, terrain_r,"
	print "PuBuGn_r, RdPu, gist_ncar_r, gist_yarg_r, Dark2_r, YlGnBu, RdYlBu, hot_r, gist_rainbow_r, gist_stern, "
	print "gnuplot_r, cool_r, cool, gray, copper_r, Greens_r, GnBu, gist_ncar, spring_r, gist_rainbow, RdYlBu_r, "
	print "gist_heat_r, OrRd_r, CMRmap, bone, gist_stern_r, RdYlGn, Pastel2_r, spring, terrain, YlOrRd_r, Set2_r, "
	print "winter_r, PuBu, RdGy_r, spectral, flag_r, jet_r, RdPu_r, Purples_r, gist_yarg, BuGn, Paired_r, hsv_r, "
	print "bwr, cubehelix, YlOrRd, Greens, PRGn, gist_heat, spectral_r, Paired, hsv, Oranges_r, prism_r, Pastel2, "
	print "Pastel1_r, Pastel1, gray_r, PuRd_r, Spectral_r, gnuplot2_r, BuPu, YlGnBu_r, copper, gist_earth_r, "
	print "Set3_r, OrRd, PuBu_r, ocean_r, brg, gnuplot2, jet, bone_r, gist_earth, Oranges, RdYlGn_r, PiYG,"
	print "CMRmap_r, YlGn, binary_r, gist_gray_r, Accent, BuPu_r, gist_gray, flag, seismic_r, RdBu_r, BrBG, Reds,"
	print "BuGn_r, summer_r, GnBu_r, BrBG_r, Reds_r, RdGy, PuRd, Accent_r, Blues, Greys, autumn, cubehelix_r, "
	print "nipy_spectral_r, PRGn_r, Greys_r, pink, binary, winter, gnuplot, RdBu, prism, YlOrBr, coolwarm_r,"
	print "rainbow_r, rainbow, PiYG_r, YlGn_r, Blues_r, YlOrBr_r, seismic, Purples, bwr_r, autumn_r, ocean,"
	print "Set1_r, PuOr, PuBuGn, nipy_spectral, afmhot."
	print ""
	sys.exit(0)

#sanity check
#============
if not os.path.isfile(args.grofilename):
	print "Error: file " + str(args.grofilename) + " not found."
	sys.exit(1)
if args.colour_file!="no" and not os.path.isfile(args.colour_file):
	print "Error: file " + str(args.colour_file) + " not found."
	sys.exit(1)
if args.selection_file_ff!="no" and not os.path.isfile(args.selection_file_ff):
	print "Error: file " + str(args.selection_file_ff) + " not found."
	sys.exit(1)
if args.xtcfilename=="no":
	if '-t' in sys.argv:
		print "Error: -t option specified but no xtc file specified."
		sys.exit(1)
	elif '-b' in sys.argv:
		print "Error: -b option specified but no xtc file specified."
		sys.exit(1)
	elif '-e' in sys.argv:
		print "Error: -e option specified but no xtc file specified."
		sys.exit(1)
	elif '--smooth' in sys.argv:
		print "Error: --smooth option specified but no xtc file specified."
		sys.exit(1)
elif not os.path.isfile(args.xtcfilename):
	print "Error: file " + str(args.xtcfilename) + " not found."
	sys.exit(1)

#create folders and log file
#===========================
if args.output_folder=="no":
	if args.xtcfilename=="no":
		args.output_folder="order_param_" + args.grofilename[:-4]
	else:
		args.output_folder="order_param_" + args.xtcfilename[:-4]
if os.path.isdir(args.output_folder):
	print "Error: folder " + str(args.output_folder) + " already exists, choose a different output name via -o."
	sys.exit(1)
else:
	#create folders
	#--------------
	os.mkdir(args.output_folder)
	#1 non flipfloppping lipids
	os.mkdir(args.output_folder + "/1_nff")
	if args.xtcfilename!="no":
		os.mkdir(args.output_folder + "/1_nff/xvg")
		os.mkdir(args.output_folder + "/1_nff/png")
		if args.nb_smoothing>1:
			os.mkdir(args.output_folder + "/1_nff/smoothed")
			os.mkdir(args.output_folder + "/1_nff/smoothed/png")
			os.mkdir(args.output_folder + "/1_nff/smoothed/xvg")
	#2 snapshots
	os.mkdir(args.output_folder + "/2_snapshots")
	#3 vmd
	if args.xtcfilename!="no":
		os.mkdir(args.output_folder + "/3_VMD")
	#4 flipflopping lipids
	if args.selection_file_ff!="no":
		os.mkdir(args.output_folder + "/4_ff")
		if args.xtcfilename!="no":
			os.mkdir(args.output_folder + "/4_ff/xvg")
			os.mkdir(args.output_folder + "/4_ff/png")
			if args.nb_smoothing>1:
				os.mkdir(args.output_folder + "/4_ff/smoothed")
				os.mkdir(args.output_folder + "/4_ff/smoothed/png")
				os.mkdir(args.output_folder + "/4_ff/smoothed/xvg")
	
	#create log
	#----------
	filename_log=os.getcwd() + '/' + str(args.output_folder) + '/order_param.log'
	output_log=open(filename_log, 'w')		
	output_log.write("[order_param v" + str(version_nb) + "]\n")
	output_log.write("\nThis folder and its content were created using the following command:\n\n")
	tmp_log="python order_param.py"
	for c in sys.argv[1:]:
		tmp_log+=" " + c
	output_log.write(tmp_log + "\n")
	output_log.close()
	#copy input files
	#----------------
	if args.colour_file!="no":
		shutil.copy2(args.colour_file,args.output_folder + "/")
	if args.selection_file_ff!="no":
		shutil.copy2(args.selection_file_ff,args.output_folder + "/")

################################################################################################################################################
# DATA LOADING
################################################################################################################################################

# Load universe
#==============
if args.xtcfilename=="no":
	print "\nLoading file..."
	U=Universe(args.grofilename)
	all_atoms=U.selectAtoms("all")
	nb_atoms=all_atoms.numberOfAtoms()
	nb_frames_xtc=1
	nb_frames_processed=1
else:
	print "\nLoading trajectory..."
	U=Universe(args.grofilename, args.xtcfilename)
	all_atoms=U.selectAtoms("all")
	nb_atoms=all_atoms.numberOfAtoms()
	nb_frames_xtc=U.trajectory.numframes
	nb_frames_processed=0
	U.trajectory.rewind()

# Identify ff lipids
#===================
lipids_ff_nb=0
lipids_ff_info={}
lipids_ff_species=[]
lipids_ff_leaflet=[]
lipids_ff_u2l_index=[]
lipids_ff_l2u_index=[]
lipids_selection_ff={}
lipids_selection_ff_VMD_string={}
leaflet_selection_string={}
leaflet_selection_string[args.forcefield_opt]="name PO4 or name PO3 or name B1A"			#martini

#case: read specified ff lipids selection file
if args.selection_file_ff!="no":
	print "\nReading selection file for flipflopping lipids..."
	with open(args.selection_file_ff) as f:
		lines = f.readlines()
	lipids_ff_nb=len(lines)
	print " -found " + str(lipids_ff_nb) + " flipflopping lipids"
	sele_all_nff_string=leaflet_selection_string[args.forcefield_opt] + " and not ("
	for l in range(0,lipids_ff_nb):
		try:
			#read the 3 comma separated field
			l_type=lines[l].split(',')[0]
			l_indx=int(lines[l].split(',')[1])
			l_start=lines[l].split(',')[2][0:-1]
			
			#build leaflet dictionary
			if l_start not in lipids_ff_leaflet:
				lipids_ff_leaflet.append(l_start)

			#create index list of u2l and l2u ff lipids
			if l_start=="upper":
				lipids_ff_u2l_index.append(l)
			elif l_start=="lower":
				lipids_ff_l2u_index.append(l)
			else:
				print "unknown starting leaflet '" + str(l_start) + "'."
				sys.exit(1)

			#build specie dictionary
			if l_type not in lipids_ff_species:
				lipids_ff_species.append(l_type)
	
			#build MDAnalysis atom group
			lipids_ff_info[l]=[l_type,l_indx,l_start]
			lipids_selection_ff[l]=U.selectAtoms("resname " + str(l_type) + " and resnum " + str(l_indx))
			if lipids_selection_ff[l].numberOfAtoms()==0:
				sys.exit(1)

			#build VMD selection string
			lipids_selection_ff_VMD_string[l]="resname " + str(lipids_ff_info[l][0]) + " and resid " + str(lipids_ff_info[l][1])
	
			#build selection string to select all PO4 without the flipflopping ones
			if l==0:
				sele_all_nff_string+="(resname " + str(l_type) + " and resnum " + str(l_indx) + ")"
			else:
				sele_all_nff_string+=" or (resname " + str(l_type) + " and resnum " + str(l_indx) + ")"
		except:
			print "Error: invalid flipflopping lipid selection string on line " + str(l+1) + ": '" + lines[l][:-1] + "'"
			sys.exit(1)
	sele_all_nff_string+=")"
#case: no ff lipids selection file specified
else:
	sele_all_nff_string=leaflet_selection_string[args.forcefield_opt]

# Identify nff leaflets
#======================
print "\nIdentifying leaflets..."
lipids_nff_sele={}
lipids_nff_sele_nb={}
for l in ["lower","upper","both"]:
	lipids_nff_sele[l]={}
	lipids_nff_sele_nb[l]={}
#identify lipids leaflet groups
if args.cutoff_leaflet:
	print " -optimising cutoff..."
	cutoff_value=MDAnalysis.analysis.leaflet.optimize_cutoff(U, sele_all_nff_string)
	L=MDAnalysis.analysis.leaflet.LeafletFinder(U, sele_all_nff_string, cutoff_value[0])
else:
	L=MDAnalysis.analysis.leaflet.LeafletFinder(U, sele_all_nff_string)
#process groups
if numpy.shape(L.groups())[0]<2:
	print "Error: imposssible to identify 2 leaflets."
	sys.exit(1)
else:
	if L.group(0).centerOfGeometry()[2] > L.group(1).centerOfGeometry()[2]:
		lipids_nff_sele["upper"]["all"]=L.group(0).residues.atoms
		lipids_nff_sele["lower"]["all"]=L.group(1).residues.atoms
	
	else:
		lipids_nff_sele["upper"]["all"]=L.group(1).residues.atoms
		lipids_nff_sele["lower"]["all"]=L.group(0).residues.atoms
	for l in ["lower","upper"]:
		lipids_nff_sele_nb[l]["all"]=lipids_nff_sele[l]["all"].numberOfResidues()
	if numpy.shape(L.groups())[0]==2:
		print " -found 2 leaflets: ", lipids_nff_sele["upper"]["all"].numberOfResidues(), '(upper) and ', lipids_nff_sele["lower"]["all"].numberOfResidues(), '(lower) lipids'
	else:
		other_lipids=0
		for g in range(2, numpy.shape(L.groups())[0]):
			other_lipids+=L.group(g).numberOfResidues()
		print " -found " + str(numpy.shape(L.groups())[0]) + " groups: " + str(lipids_nff_sele["upper"]["all"].numberOfResidues()) + "(upper), " + str(lipids_nff_sele["lower"]["all"].numberOfResidues()) + "(lower) and " + str(other_lipids) + " (others) lipids respectively"
lipids_nff_sele["both"]["all"]=lipids_nff_sele["lower"]["all"]+lipids_nff_sele["upper"]["all"]
lipids_nff_sele_nb_atoms=lipids_nff_sele["both"]["all"].numberOfAtoms()

# Identify lipid species
#=======================
print "\nIdentifying membrane composition..."
#specie identification
lipids_nff_species={}
for l in ["lower","upper"]:
	lipids_nff_species[l]=list(numpy.unique(lipids_nff_sele[l]["all"].resnames()))
lipids_nff_species["both"]=numpy.unique(lipids_nff_species["lower"]+lipids_nff_species["upper"])

#count species
lipids_nff_species_nb={}
for l in ["lower","upper","both"]:
	lipids_nff_species_nb[l]=numpy.size(lipids_nff_species[l])

#selection creation
for l in ["lower","upper"]:
	for s in lipids_nff_species[l]:		
		#current specie
		lipids_nff_sele[l][s]=lipids_nff_sele[l]["all"].selectAtoms("resname " + str(s))
		lipids_nff_sele_nb[l][s]=lipids_nff_sele[l][s].numberOfResidues()

#specie ratios
membrane_comp={}
lipids_nff_ratio={}
for l in ["lower","upper"]:
	membrane_comp[l]=" -" + str(l) + ":"
	lipids_nff_ratio[l]={}
	for s in lipids_nff_species[l]:
		lipids_nff_ratio[l][s]=round(lipids_nff_sele_nb[l][s]/float(lipids_nff_sele_nb[l]["all"])*100,1)
		membrane_comp[l]+=" " + s + " (" + str(lipids_nff_ratio[l][s]) + "%)"
	print membrane_comp[l]

################################################################################################################################################
# LIPIDS DICTIONARIES
################################################################################################################################################

#define all lipids taken into account
#====================================
#lipids handled
lipids_possible={}
lipids_possible[args.forcefield_opt]=['DHPC','DHPE','DLPC','DLPE','DAPC','DUPC','DPPC','DPPE','DPPS','DPPG','DSPC','DSPE','POPC','POPE','POPS','POPG','PPCS','PIP2','PIP3','GM3']

#case: martini
#-------------
if args.forcefield_opt=="martini":

	#define possible head, link and tail sequences
	head_PC=" NC3-PO4"
	head_PE=" NH3-PO4"
	head_PS=" CNO-PO4"
	head_PG=" GL0-PO4"
	head_I2=" PO1-RP1 PO2-RP1 PO2-RP2 RP1-RP2 RP1-RP3 RP2-RP3 RP3-PO3"
	head_I3=" PO0-RP2 PO1-RP1 PO2-RP1 PO2-RP2 RP1-RP2 RP1-RP3 RP2-RP3 RP3-PO3"
	head_GM=" very-big"
	link1=" PO4-GL1 GL1-GL2 GL1-C1A GL2-C1B"
	link2=" PO4-GL1 GL1-GL2 GL1-D1A GL2-D1B"
	link3=" PO4-AM1 AM1-AM2 AM1-C1A AM2-D1B"
	link4=" PO3-GL1 GL1-GL2 GL1-C1A GL2-C1B"
	link5=" B1A-AM2 AM1-AM2 AM1-C1A AM2-D1B"
	tail_DH=" C1A-C2A C1B-C2B"
	tail_DL=" C1A-C2A C2A-C3A C1B-C2B C2B-C3B"
	tail_DP=" C1A-C2A C2A-C3A C3A-C4A C1B-C2B C2B-C3B C3B-C4B"
	tail_DU=" C1A-D2A D2A-D3A D3A-C4A C1B-D2B D2B-D3B D3B-C4B"
	tail_DS=" C1A-C2A C2A-C3A C3A-C4A C4A-C5A C1B-C2B C2B-C3B C3B-C4B C4B-C5B"
	tail_DO=" C1A-C2A C2A-D3A D3A-C4A C4A-C5A C1B-C2B C2B-D3B D3B-C4B C4B-C5B"
	tail_DA=" D1A-D2A D2A-D3A D3A-D4A D4A-C5A D1B-D2B D2B-D3B D3B-D4B D4B-C5B"
	tail_PO=" C1A-C2A C2A-C3A C3A-C4A C1B-C2B C2B-D3B D3B-C4B C4B-C5B"
	tail_PI=" C1A-C2A C2A-C3A C3A-C4A C1B-C2B C2B-C3B C3B-C4B C4B-C5B"
	tail_PP=" C1A-C2A C2A-C3A C3A-C4A D1B-C2B C2B-C3B C3B-C4B"
	
	#define lipids composition
	bond_names={}	
	bond_names['DHPC']=head_PC + link1 + tail_DH
	bond_names['DHPE']=head_PE + link1 + tail_DH
	bond_names['DLPC']=head_PC + link1 + tail_DL
	bond_names['DLPE']=head_PE + link1 + tail_DL
	bond_names['DAPC']=head_PC + link2 + tail_DA
	bond_names['DUPC']=head_PC + link1 + tail_DU
	bond_names['DPPC']=head_PC + link1 + tail_DP
	bond_names['DPPE']=head_PE + link1 + tail_DP
	bond_names['DPPS']=head_PS + link1 + tail_DP
	bond_names['DPPG']=head_PG + link1 + tail_DP
	bond_names['DSPC']=head_PC + link1 + tail_DS
	bond_names['DSPE']=head_PE + link1 + tail_DS
	bond_names['POPC']=head_PC + link1 + tail_PO
	bond_names['POPE']=head_PE + link1 + tail_PO
	bond_names['POPS']=head_PS + link1 + tail_PO
	bond_names['POPG']=head_PG + link1 + tail_PO
	bond_names['PPCS']=head_PC + link3 + tail_PP
	bond_names['PIP2']=head_I2 + link4 + tail_PI
	bond_names['PIP3']=head_I3 + link4 + tail_PI
	bond_names['GM3']=head_GM + link5 + tail_PP
	
	#define tail boundaries (start_A, length_A, start_B, length_B)
	tail_boundaries={}
	tail_boundaries['DHPC']=[5,1,6,1]
	tail_boundaries['DHPE']=[5,1,6,1]	
	tail_boundaries['DLPC']=[5,2,7,2]
	tail_boundaries['DLPE']=[5,2,7,2]	
	tail_boundaries['DAPC']=[5,4,9,4]
	tail_boundaries['DUPC']=[5,4,9,4]	
	tail_boundaries['DPPC']=[5,3,8,3]
	tail_boundaries['DPPG']=[5,3,8,3]	
	tail_boundaries['DPPE']=[5,3,8,3]	
	tail_boundaries['DPPS']=[5,3,8,3]		
	tail_boundaries['DOPC']=[5,4,9,4]
	tail_boundaries['DOPG']=[5,4,9,4]	
	tail_boundaries['DOPE']=[5,4,9,4]	
	tail_boundaries['DOPS']=[5,4,9,4]		
	tail_boundaries['DSPC']=[5,4,9,4]
	tail_boundaries['DSPE']=[5,4,9,4]
	tail_boundaries['POPC']=[5,3,8,4]
	tail_boundaries['POPE']=[5,3,8,4]
	tail_boundaries['POPS']=[5,3,8,4]
	tail_boundaries['POPG']=[5,3,8,4]
	tail_boundaries['PPCS']=[5,3,8,3]	
	tail_boundaries['PIP2']=[11,3,14,4]
	tail_boundaries['PIP3']=[12,3,15,4]
	tail_boundaries['GM3']=[5,3,8,3]

#deal with those actually present
#================================
#create list of lipids to take into account
#------------------------------------------
lipids_handled={}
for l in ["lower","upper"]:
	lipids_handled[l]=[]
	for s in lipids_nff_species[l]:
		if s in lipids_possible[args.forcefield_opt]:
			lipids_handled[l].append(s)
if len(lipids_handled["lower"])==0 and len(lipids_handled["upper"])==0:
	print "Error: none of the lipid species can be taken into account - double check the forcefield option, see order_param -h."
	sys.exit(1)
lipids_handled["both"]=numpy.unique(lipids_handled["lower"]+lipids_handled["upper"])

#display them
#------------
tmp_lip=lipids_handled["both"][0]
for s in lipids_handled["both"][1:]:
	tmp_lip+=","+str(s)
print "\nLipids handled: ", tmp_lip

print "\nInitialising data structures..."

#create VMD selection string for each specie
#-------------------------------------------
lipids_selection_nff={}
lipids_selection_nff_VMD_string={}
for l in ["lower","upper"]:
	lipids_selection_nff[l]={}
	lipids_selection_nff_VMD_string[l]={}
	for s in lipids_handled[l]:
		lipids_selection_nff[l][s]={}
		lipids_selection_nff_VMD_string[l][s]={}
		for r_index in range(0,lipids_nff_sele_nb[l][s]):
			lipids_selection_nff[l][s][r_index]=lipids_nff_sele[l][s].selectAtoms("resnum " + str(lipids_nff_sele[l][s].resnums()[r_index]))
			lipids_selection_nff_VMD_string[l][s][r_index]="resname " + str(s) + " and resid " + str(lipids_nff_sele[l][s].resnums()[r_index])

#create bond list for each lipid specie
#--------------------------------------
bonds={}
for s in lipids_handled["both"]:
	bonds[s]=[]
	for bond_name in bond_names[s].split():
		bonds[s].append(bond_name.split("-"))

#associate colours to lipids
#===========================
#color maps dictionaries
colours_lipids_nb=0
colours_lipids={}
colours_lipids_list=[]
colours_lipids_map="jet"
colormaps_possible=['Spectral', 'summer', 'coolwarm', 'pink_r', 'Set1', 'Set2', 'Set3', 'brg_r', 'Dark2', 'hot', 'PuOr_r', 'afmhot_r', 'terrain_r', 'PuBuGn_r', 'RdPu', 'gist_ncar_r', 'gist_yarg_r', 'Dark2_r', 'YlGnBu', 'RdYlBu', 'hot_r', 'gist_rainbow_r', 'gist_stern', 'gnuplot_r', 'cool_r', 'cool', 'gray', 'copper_r', 'Greens_r', 'GnBu', 'gist_ncar', 'spring_r', 'gist_rainbow', 'RdYlBu_r', 'gist_heat_r', 'OrRd_r', 'CMRmap', 'bone', 'gist_stern_r', 'RdYlGn', 'Pastel2_r', 'spring', 'terrain', 'YlOrRd_r', 'Set2_r', 'winter_r', 'PuBu', 'RdGy_r', 'spectral', 'flag_r', 'jet_r', 'RdPu_r', 'Purples_r', 'gist_yarg', 'BuGn', 'Paired_r', 'hsv_r', 'bwr', 'cubehelix', 'YlOrRd', 'Greens', 'PRGn', 'gist_heat', 'spectral_r', 'Paired', 'hsv', 'Oranges_r', 'prism_r', 'Pastel2', 'Pastel1_r', 'Pastel1', 'gray_r', 'PuRd_r', 'Spectral_r', 'gnuplot2_r', 'BuPu', 'YlGnBu_r', 'copper', 'gist_earth_r', 'Set3_r', 'OrRd', 'PuBu_r', 'ocean_r', 'brg', 'gnuplot2', 'jet', 'bone_r', 'gist_earth', 'Oranges', 'RdYlGn_r', 'PiYG', 'CMRmap_r', 'YlGn', 'binary_r', 'gist_gray_r', 'Accent', 'BuPu_r', 'gist_gray', 'flag', 'seismic_r', 'RdBu_r', 'BrBG', 'Reds', 'BuGn_r', 'summer_r', 'GnBu_r', 'BrBG_r', 'Reds_r', 'RdGy', 'PuRd', 'Accent_r', 'Blues', 'Greys', 'autumn', 'cubehelix_r', 'nipy_spectral_r', 'PRGn_r', 'Greys_r', 'pink', 'binary', 'winter', 'gnuplot', 'RdBu', 'prism', 'YlOrBr', 'coolwarm_r', 'rainbow_r', 'rainbow', 'PiYG_r', 'YlGn_r', 'Blues_r', 'YlOrBr_r', 'seismic', 'Purples', 'bwr_r', 'autumn_r', 'ocean', 'Set1_r', 'PuOr', 'PuBuGn', 'nipy_spectral', 'afmhot']
#case: group definition file
#---------------------------
if args.colour_file!="no":
	
	print "\nReading colour definition file..."
	with open(args.colour_file) as f:
		lines = f.readlines()
	colours_lipids_nb=len(lines)
	for line_index in range(0,colours_lipids_nb):
		l_content=lines[line_index].split(',')
		colours_lipids[l_content[0]]=l_content[1][:-1]					#to get rid of the returning char
	
	#display results
	print " -found the following colours definition:"
	for s in colours_lipids.keys():
		print " -" + str(s) + ": " + str(colours_lipids[s])

	#check if a custom color map has been specified or not
	if colours_lipids_nb>1 and len(numpy.unique(colours_lipids.values()))==1:
		if numpy.unique(colours_lipids.values())[0] in colormaps_possible:
			colours_lipids_map=numpy.unique(colours_lipids.values())[0]
		else:
			print "Error: either the same color was specified for all species or the color map '" + str(numpy.unique(colours_lipids.values())[0]) + "' is not valid."
			sys.exit(1)
	else:
		colours_lipids_map="custom"
		
	#check that all detected species have a colour specified
	for s in lipids_handled["both"]:
		if s not in colours_lipids.keys():
			print "Error: no colour specified for " + str(s) + "."
			sys.exit(1)

#case: generate colours from jet colour map
#------------------------------------------
if colours_lipids_map!="custom":
	tmp_cmap=cm.get_cmap(colours_lipids_map)
	colours_lipids_value=tmp_cmap(numpy.linspace(0, 1, len(lipids_handled["both"])))
	for l_index in range(0, len(lipids_handled["both"])):
		colours_lipids[lipids_handled["both"][l_index]]=colours_lipids_value[l_index]


################################################################################################################################################
# DATA STRUCTURE: order parameters
################################################################################################################################################

#time
#----
time_stamp={}
time_sorted=[]
time_smooth=[]

#non flipflopping lipids
#-----------------------
#avg over lipids within a frame: full data
op_tailA_avg_frame={}
op_tailB_avg_frame={}
op_both_avg_frame={}
op_tailA_std_frame={}
op_tailB_std_frame={}
op_both_std_frame={}
#avg over lipids within a frame: sorted data
op_tailA_avg_frame_sorted={}
op_tailB_avg_frame_sorted={}
op_both_avg_frame_sorted={}
op_tailA_std_frame_sorted={}
op_tailB_std_frame_sorted={}
op_both_std_frame_sorted={}
for l in ["lower","upper"]:
	op_tailA_avg_frame_sorted[l]={}
	op_tailB_avg_frame_sorted[l]={}			
	op_both_avg_frame_sorted[l]={}
	op_tailA_std_frame_sorted[l]={}
	op_tailB_std_frame_sorted[l]={}
	op_both_std_frame_sorted[l]={}
	for s in lipids_handled[l]:
		op_tailA_avg_frame_sorted[l][s]=[]
		op_tailB_avg_frame_sorted[l][s]=[]			
		op_both_avg_frame_sorted[l][s]=[]
		op_tailA_std_frame_sorted[l][s]=[]
		op_tailB_std_frame_sorted[l][s]=[]
		op_both_std_frame_sorted[l][s]=[]
#avg over lipids within a frame: smoothed data
op_tailA_avg_frame_smooth={}
op_tailB_avg_frame_smooth={}
op_both_avg_frame_smooth={}
op_tailA_std_frame_smooth={}
op_tailB_std_frame_smooth={}
op_both_std_frame_smooth={}
for l in ["lower","upper"]:
	op_tailA_avg_frame_smooth[l]={}
	op_tailB_avg_frame_smooth[l]={}			
	op_both_avg_frame_smooth[l]={}
	op_tailA_std_frame_smooth[l]={}
	op_tailB_std_frame_smooth[l]={}
	op_both_std_frame_smooth[l]={}
#avg over time of frame avg
op_tailA_avg={}
op_tailB_avg={}
op_both_avg={}
op_tailA_std={}
op_tailB_std={}
op_both_std={}
for l in ["lower","upper"]:
	op_tailA_avg[l]={}
	op_tailB_avg[l]={}
	op_both_avg[l]={}
	op_tailA_std[l]={}
	op_tailB_std[l]={}
	op_both_std[l]={}
	op_tailA_avg_frame[l]={}
	op_tailB_avg_frame[l]={}			
	op_both_avg_frame[l]={}
	op_tailA_std_frame[l]={}
	op_tailB_std_frame[l]={}
	op_both_std_frame[l]={}
	for s in lipids_handled[l]:
		op_tailA_avg_frame[l][s]={}
		op_tailB_avg_frame[l][s]={}			
		op_both_avg_frame[l][s]={}
		op_tailA_std_frame[l][s]={}
		op_tailB_std_frame[l][s]={}
		op_both_std_frame[l][s]={}
#store evolution of op for each lipid
lipids_nff_op={}
for l in ["lower","upper"]:
	lipids_nff_op[l]={}
	for s in lipids_handled[l]:
		lipids_nff_op[l][s]={}
		for r_index in range(0,lipids_nff_sele_nb[l][s]):
			lipids_nff_op[l][s][r_index]=[]

#flipflopping lipids
#-------------------
#order parameter: full data
op_ff_tailA={}
op_ff_tailB={}
op_ff_both={}
for l in range(0,lipids_ff_nb):
	op_ff_tailA[l]={}
	op_ff_tailB[l]={}
	op_ff_both[l]={}
#order paramater: sorted data
op_ff_tailA_sorted={}
op_ff_tailB_sorted={}
op_ff_both_sorted={}
for l in range(0,lipids_ff_nb):
	op_ff_tailA_sorted[l]=[]
	op_ff_tailB_sorted[l]=[]
	op_ff_both_sorted[l]=[]
#order paramater: smoothed data
op_ff_tailA_smooth={}
op_ff_tailB_smooth={}
op_ff_both_smooth={}

#z coordinate: full data
z_lower={}																#store z coord of lower leaflet for each frame
z_upper={}																#store z coord of upper leaflet for each frame
z_ff={}																	#store z coord of the PO4 particle of each ff lipid
for l in range(0,lipids_ff_nb):
	z_ff[l]={}
#z coordinate: sorted data
z_upper_sorted=[]
z_lower_sorted=[]
z_ff_sorted={}
for l in range(0,lipids_ff_nb):
	z_ff_sorted[l]=[]
#z coordinate: smoothed data
z_lower_smooth=[]
z_upper_smooth=[]
z_ff_smooth={}

################################################################################################################################################
# FUNCTIONS: core
################################################################################################################################################

def get_z_coords(frame_nb):

	z_middle_instant=lipids_nff_sele["lower"]["all"].selectAtoms("name PO4").centerOfGeometry()[2]+(lipids_nff_sele["upper"]["all"].selectAtoms("name PO4").centerOfGeometry()[2]-lipids_nff_sele["lower"]["all"].selectAtoms("name PO4").centerOfGeometry()[2])/float(2)
	z_lower[frame_nb]=lipids_nff_sele["lower"]["all"].selectAtoms("name PO4").centerOfGeometry()[2]-z_middle_instant
	z_upper[frame_nb]=lipids_nff_sele["upper"]["all"].selectAtoms("name PO4").centerOfGeometry()[2]-z_middle_instant
	for l in range(0,lipids_ff_nb):	
		z_ff[l][frame_nb]=lipids_selection_ff[l].selectAtoms("name PO4").centerOfGeometry()[2]-z_middle_instant

	return
def calculate_order_parameters(frame_nb):
	
	#non flipflopping lipids
	#=======================
	for l in ["lower","upper"]:		
		for s in lipids_handled[l]:
			#retrieve tail boundaries for current lipid type
			tail_A_start=tail_boundaries[s][0]
			tail_B_start=tail_boundaries[s][2]
			tail_A_length=tail_boundaries[s][1]
			tail_B_length=tail_boundaries[s][3]

			#retrieve coordinates of lipids
			tmp_coord=lipids_nff_sele[l][s].coordinates()
			tmp_op=[]
			
			#calculate 'order param' for each bond (1st bond to initialise array)
			tmp_bond_array=numpy.zeros((lipids_nff_sele[l][s].numberOfResidues(),1))
			v=numpy.zeros((lipids_nff_sele[l][s].numberOfResidues(),3))
			v_norm2=numpy.zeros((lipids_nff_sele[l][s].numberOfResidues(),1))			
			v[:,0]=lipids_nff_sele[l][s].selectAtoms("name " + str(bonds[s][tail_boundaries[s][0]][0])).coordinates()[:,0]-lipids_nff_sele[l][s].selectAtoms("name " + str(bonds[s][tail_boundaries[s][0]][1])).coordinates()[:,0]
			v[:,1]=lipids_nff_sele[l][s].selectAtoms("name " + str(bonds[s][tail_boundaries[s][0]][0])).coordinates()[:,1]-lipids_nff_sele[l][s].selectAtoms("name " + str(bonds[s][tail_boundaries[s][0]][1])).coordinates()[:,1]
			v[:,2]=lipids_nff_sele[l][s].selectAtoms("name " + str(bonds[s][tail_boundaries[s][0]][0])).coordinates()[:,2]-lipids_nff_sele[l][s].selectAtoms("name " + str(bonds[s][tail_boundaries[s][0]][1])).coordinates()[:,2]
			v_norm2[:,0] = v[:,0]**2 + v[:,1]**2 + v[:,2]**2
			tmp_bond_array[:,0]=0.5*(3*(v[:,2]**2)/v_norm2[:,0]-1)
			for bond in bonds[s][tail_boundaries[s][0]+1:]:
				v=numpy.zeros((lipids_nff_sele[l][s].numberOfResidues(),3))
				v_norm2=numpy.zeros((lipids_nff_sele[l][s].numberOfResidues(),1))
				tmp_op_bond=numpy.zeros((lipids_nff_sele[l][s].numberOfResidues(),1))
				v[:,0]=lipids_nff_sele[l][s].selectAtoms("name " + str(bond[0])).coordinates()[:,0]-lipids_nff_sele[l][s].selectAtoms("name " + str(bond[1])).coordinates()[:,0]
				v[:,1]=lipids_nff_sele[l][s].selectAtoms("name " + str(bond[0])).coordinates()[:,1]-lipids_nff_sele[l][s].selectAtoms("name " + str(bond[1])).coordinates()[:,1]
				v[:,2]=lipids_nff_sele[l][s].selectAtoms("name " + str(bond[0])).coordinates()[:,2]-lipids_nff_sele[l][s].selectAtoms("name " + str(bond[1])).coordinates()[:,2]
				v_norm2[:,0] = v[:,0]**2 + v[:,1]**2 + v[:,2]**2
				tmp_op_bond[:,0]=0.5*(3*(v[:,2]**2)/v_norm2[:,0]-1)
				tmp_bond_array=numpy.concatenate((tmp_bond_array, tmp_op_bond), axis=1)

			#calculate op (tail A, tail B, avg of both) for each lipid
			tmp_op_tails_array=numpy.zeros((lipids_nff_sele[l][s].numberOfResidues(),2))
			tmp_op_tails_array[:,0]=numpy.average(tmp_bond_array[:,0:tail_A_length],axis=1)
			tmp_op_tails_array[:,1]=numpy.average(tmp_bond_array[:,tail_A_length:tail_A_length+tail_B_length],axis=1)
			tmp_op_array=numpy.average(tmp_op_tails_array, axis=1)
						
			#calculate averages for whole lipid specie
			op_tailA_avg_frame[l][s][frame_nb]=numpy.average(tmp_op_tails_array[:,0])
			op_tailB_avg_frame[l][s][frame_nb]=numpy.average(tmp_op_tails_array[:,1])
			op_both_avg_frame[l][s][frame_nb]=numpy.average(tmp_op_array)
			op_tailA_std_frame[l][s][frame_nb]=numpy.std(tmp_op_tails_array[:,0])
			op_tailB_std_frame[l][s][frame_nb]=numpy.std(tmp_op_tails_array[:,1])
			op_both_std_frame[l][s][frame_nb]=numpy.std(tmp_op_array)			

			#store order parameter for each residue
			for r_index in range(0,lipids_nff_sele_nb[l][s]):
				lipids_nff_op[l][s][r_index].append(tmp_op_array[r_index])

			
	#flipflopping lipids
	#===================
	if args.selection_file_ff!="no":
		for l in range(0, lipids_ff_nb):
			tmp_bond_array=[]
			tmp_specie=lipids_ff_info[l][0]
			for bond in bonds[tmp_specie][tail_boundaries[tmp_specie][0]:]:
				vx=lipids_selection_ff[l].selectAtoms("name " + str(bond[0])).coordinates()[0,0]-lipids_selection_ff[l].selectAtoms("name " + str(bond[1])).coordinates()[0,0]
				vy=lipids_selection_ff[l].selectAtoms("name " + str(bond[0])).coordinates()[0,1]-lipids_selection_ff[l].selectAtoms("name " + str(bond[1])).coordinates()[0,1]
				vz=lipids_selection_ff[l].selectAtoms("name " + str(bond[0])).coordinates()[0,2]-lipids_selection_ff[l].selectAtoms("name " + str(bond[1])).coordinates()[0,2]
				v_norm2=vx**2 + vy**2 + vz**2
				tmp_bond_array.append(0.5*(3*(vz**2)/float(v_norm2)-1))
			op_ff_tailA[l][frame_nb]=numpy.average(tmp_bond_array[0:tail_A_length])
			op_ff_tailB[l][frame_nb]=numpy.average(tmp_bond_array[tail_A_length:tail_A_length+tail_B_length])
			op_ff_both[l][frame_nb]=numpy.average(op_ff_tailA[l][frame_nb], op_ff_tailB[l][frame_nb])

	return
def rolling_avg(loc_list):
	
	loc_arr=numpy.asarray(loc_list)
	shape=(loc_arr.shape[-1]-args.nb_smoothing+1,args.nb_smoothing)
	strides=(loc_arr.strides[-1],loc_arr.strides[-1])   	
	return numpy.average(numpy.lib.stride_tricks.as_strided(loc_arr, shape=shape, strides=strides), -1)
def calculate_stats():
	
	for l in ["lower","upper"]:
		#define data structure
		#---------------------
		op_tailA_avg[l]["all"]=[]
		op_tailA_std[l]["all"]=[]
		op_tailB_avg[l]["all"]=[]
		op_tailB_std[l]["all"]=[]
		op_both_avg[l]["all"]=[]
		op_both_std[l]["all"]=[]
	
		#store specie average
		#--------------------
		#case: gro file
		if args.xtcfilename=="no":
			for s in lipids_handled[l]:
				op_tailA_avg[l][s]=op_tailA_avg_frame[l][s][1]
				op_tailA_avg[l]["all"].append(op_tailA_avg_frame[l][s][1]*lipids_nff_sele[l][s].numberOfResidues())
				op_tailB_avg[l][s]=op_tailB_avg_frame[l][s][1]
				op_tailB_avg[l]["all"].append(op_tailB_avg_frame[l][s][1]*lipids_nff_sele[l][s].numberOfResidues())
				op_both_avg[l][s]=op_both_avg_frame[l][s][1]
				op_both_avg[l]["all"].append(op_both_avg_frame[l][s][1]*lipids_nff_sele[l][s].numberOfResidues())
				op_tailA_std[l][s]=op_tailA_std_frame[l][s][1]
				op_tailA_std[l]["all"].append(op_tailA_std_frame[l][s][1]*lipids_nff_sele[l][s].numberOfResidues())
				op_tailB_std[l][s]=op_tailB_std_frame[l][s][1]
				op_tailB_std[l]["all"].append(op_tailB_std_frame[l][s][1]*lipids_nff_sele[l][s].numberOfResidues())
				op_both_std[l][s]=op_both_std_frame[l][s][1]
				op_both_std[l]["all"].append(op_both_std_frame[l][s][1]*lipids_nff_sele[l][s].numberOfResidues())
		#case: xtc file
		else:
			for s in lipids_handled[l]:
				op_tailA_avg[l][s]=numpy.average(op_tailA_avg_frame[l][s].values())
				op_tailA_avg[l]["all"].append(numpy.average(op_tailA_avg_frame[l][s].values())*lipids_nff_sele[l][s].numberOfResidues())
				op_tailB_avg[l][s]=numpy.average(op_tailB_avg_frame[l][s].values())
				op_tailB_avg[l]["all"].append(numpy.average(op_tailB_avg_frame[l][s].values())*lipids_nff_sele[l][s].numberOfResidues())
				op_both_avg[l][s]=numpy.average(op_both_avg_frame[l][s].values())
				op_both_avg[l]["all"].append(numpy.average(op_both_avg_frame[l][s].values())*lipids_nff_sele[l][s].numberOfResidues())
				op_tailA_std[l][s]=numpy.average(op_tailA_std_frame[l][s].values())
				op_tailA_std[l]["all"].append(numpy.average(op_tailA_std_frame[l][s].values())*lipids_nff_sele[l][s].numberOfResidues())
				op_tailB_std[l][s]=numpy.average(op_tailB_std_frame[l][s].values())
				op_tailB_std[l]["all"].append(numpy.average(op_tailB_std_frame[l][s].values())*lipids_nff_sele[l][s].numberOfResidues())
				op_both_std[l][s]=numpy.average(op_both_std_frame[l][s].values())
				op_both_std[l]["all"].append(numpy.average(op_both_std_frame[l][s].values())*lipids_nff_sele[l][s].numberOfResidues())
	
		#calculate leaflet average
		#-------------------------
		op_tailA_avg[l]["all"]=numpy.sum(op_tailA_avg[l]["all"])/float(lipids_nff_sele_nb["upper"]["all"])
		op_tailB_avg[l]["all"]=numpy.sum(op_tailB_avg[l]["all"])/float(lipids_nff_sele_nb["upper"]["all"])	
		op_both_avg[l]["all"]=numpy.sum(op_both_avg[l]["all"])/float(lipids_nff_sele_nb["upper"]["all"])
		op_tailA_std[l]["all"]=numpy.sum(op_tailA_std[l]["all"])/float(lipids_nff_sele_nb["upper"]["all"])
		op_tailB_std[l]["all"]=numpy.sum(op_tailB_std[l]["all"])/float(lipids_nff_sele_nb["upper"]["all"])	
		op_both_std[l]["all"]=numpy.sum(op_both_std[l]["all"])/float(lipids_nff_sele_nb["upper"]["all"])
		
	return
def smooth_data():
	
	global time_smooth
	global z_upper_smooth
	global z_lower_smooth
	
	#time and nff lipids
	#===================
	#sort data into ordered lists
	for frame in sorted(time_stamp.keys()):
		time_sorted.append(time_stamp[frame])
		for l in ["lower","upper"]:
			for s in lipids_handled[l]:
				op_tailA_avg_frame_sorted[l][s].append(op_tailA_avg_frame[l][s][frame])
				op_tailB_avg_frame_sorted[l][s].append(op_tailB_avg_frame[l][s][frame])
				op_both_avg_frame_sorted[l][s].append(op_both_avg_frame[l][s][frame])
				op_tailA_std_frame_sorted[l][s].append(op_tailA_std_frame[l][s][frame])
				op_tailB_std_frame_sorted[l][s].append(op_tailB_std_frame[l][s][frame])
				op_both_std_frame_sorted[l][s].append(op_both_std_frame[l][s][frame])
	
	#calculate running average on sorted lists
	time_smooth=rolling_avg(time_sorted)	
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			op_tailA_avg_frame_smooth[l][s]=rolling_avg(op_tailA_avg_frame_sorted[l][s])
			op_tailB_avg_frame_smooth[l][s]=rolling_avg(op_tailB_avg_frame_sorted[l][s])
			op_both_avg_frame_smooth[l][s]=rolling_avg(op_both_avg_frame_sorted[l][s])
			op_tailA_std_frame_smooth[l][s]=rolling_avg(op_tailA_std_frame_sorted[l][s])
			op_tailB_std_frame_smooth[l][s]=rolling_avg(op_tailB_std_frame_sorted[l][s])
			op_both_std_frame_smooth[l][s]=rolling_avg(op_both_std_frame_sorted[l][s])

	
	#flipflopping lipids
	#===================
	if args.selection_file_ff!="no":
		#sort data into ordered lists
		for frame in sorted(time_stamp.keys()):		
			z_upper_sorted.append(z_upper[frame])
			z_lower_sorted.append(z_lower[frame])
			for l in range(0,lipids_ff_nb):
				z_ff_sorted[l].append(z_ff[l][frame])
				op_ff_tailA_sorted[l].append(op_ff_tailA[l][frame])
				op_ff_tailB_sorted[l].append(op_ff_tailB[l][frame])
				op_ff_both_sorted[l].append(op_ff_both[l][frame])
	
		#calculate running average on sorted lists
		z_upper_smooth=rolling_avg(z_upper_sorted)
		z_lower_smooth=rolling_avg(z_lower_sorted)
		for l in range(0,lipids_ff_nb):
			z_ff_smooth[l]=rolling_avg(z_ff_sorted[l])
			op_ff_tailA_smooth[l]=rolling_avg(op_ff_tailA_sorted[l])
			op_ff_tailB_smooth[l]=rolling_avg(op_ff_tailB_sorted[l])
			op_ff_both_smooth[l]=rolling_avg(op_ff_both_sorted[l])

	return

################################################################################################################################################
# FUNCTIONS: outputs
################################################################################################################################################

#non flipflopping lipids
def write_op_nff_xvg():
	
	#lipids in upper leaflet
	#=======================
	filename_txt=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/xvg/1_3_order_param_nff_upper.txt'
	filename_xvg=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/xvg/1_3_order_param_nff_upper.xvg'
	output_txt = open(filename_txt, 'w')
	output_txt.write("@[lipid tail order parameters statistics - written by order_param v" + str(version_nb) + "]\n")
	output_txt.write("@Use this file as the argument of the -c option of the script 'xvg_animate' in order to make a time lapse movie of the data in 1_3_order_param_nff_upper.xvg.\n")
	output_xvg = open(filename_xvg, 'w')
	output_xvg.write("@ title \"Evolution of lipid tails order parameters in upper leaflet\n")
	output_xvg.write("@ xaxis  label \"time (ns)\"\n")
	output_xvg.write("@ yaxis  label \"order parameter S2\"\n")
	output_xvg.write("@ autoscale ONREAD xaxes\n")
	output_xvg.write("@ TYPE XY\n")
	output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
	output_xvg.write("@ legend on\n")
	output_xvg.write("@ legend box on\n")
	output_xvg.write("@ legend loctype view\n")
	output_xvg.write("@ legend 0.98, 0.8\n")
	output_xvg.write("@ legend length " + str(2*len(lipids_handled["upper"])*3) + "\n")
	for s_index in range(0,len(lipids_handled["upper"])):
		output_xvg.write("@ s" + str(3*s_index) + " legend \"" + str(lipids_handled["upper"][s_index]) + " tail A (avg)\"\n")
		output_xvg.write("@ s" + str(3*s_index+1) + " legend \"" + str(lipids_handled["upper"][s_index]) + " tail B (avg)\"\n")
		output_xvg.write("@ s" + str(3*s_index+2) + " legend \"" + str(lipids_handled["upper"][s_index]) + " both (avg)\"\n")
		output_txt.write("1_3_order_param_nff_upper.xvg," + str((3*s_index)+1) +"," + str(lipids_handled["upper"][s_index]) + " tail A (avg)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["upper"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_upper.xvg," + str((3*s_index+1)+1) +"," + str(lipids_handled["upper"][s_index]) + " tail B (avg)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["upper"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_upper.xvg," + str((3*s_index+2)+1) +"," + str(lipids_handled["upper"][s_index]) + " both (avg)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["upper"][s_index])]) + "\n")
	for s_index in range(0,len(lipids_handled["upper"])):
		output_xvg.write("@ s" + str(3*len(lipids_handled["upper"])+3*s_index) + " legend \"" + str(lipids_handled["upper"][s_index]) + " tail A (std)\"\n")
		output_xvg.write("@ s" + str(3*len(lipids_handled["upper"])+3*s_index+1) + " legend \"" + str(lipids_handled["upper"][s_index]) + " tail (std)B\"\n")
		output_xvg.write("@ s" + str(3*len(lipids_handled["upper"])+3*s_index+2) + " legend \"" + str(lipids_handled["upper"][s_index]) + " both (std)\"\n")
		output_txt.write("1_3_order_param_nff_upper.xvg," + str(3*len(lipids_handled["upper"])+(3*s_index)+1) +"," + str(lipids_handled["upper"][s_index]) + " tail A (std)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["upper"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_upper.xvg," + str(3*len(lipids_handled["upper"])+(3*s_index+1)+1) +"," + str(lipids_handled["upper"][s_index]) + " tail B (std)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["upper"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_upper.xvg," + str(3*len(lipids_handled["upper"])+(3*s_index+2)+1) +"," + str(lipids_handled["upper"][s_index]) + " both (std)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["upper"][s_index])]) + "\n")
	output_txt.close()
	for frame in sorted(time_stamp.iterkeys()):
		results=str(time_stamp[frame])
		for s in lipids_handled["upper"]:
			results+="	" + str(round(op_tailA_avg_frame["upper"][s][frame],2)) + "	" + str(round(op_tailB_avg_frame["upper"][s][frame],2)) + "	" + str(round(op_both_avg_frame["upper"][s][frame],2))
		for s in lipids_handled["upper"]:
			results+="	" + str(round(op_tailA_std_frame["upper"][s][frame],2)) + "	" + str(round(op_tailB_std_frame["upper"][s][frame],2)) + "	" + str(round(op_both_std_frame["upper"][s][frame],2))
		output_xvg.write(results + "\n")
	output_xvg.close()

	#lipids in lower leaflet
	#=======================
	filename_txt=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/xvg/1_3_order_param_nff_lower.txt'
	filename_xvg=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/xvg/1_3_order_param_nff_lower.xvg'
	output_txt = open(filename_txt, 'w')
	output_txt.write("@[lipid tail order parameters statistics - written by order_param v" + str(version_nb) + "]\n")
	output_txt.write("@Use this file as the argument of the -c option of the script 'xvg_animate' in order to make a time lapse movie of the data in 1_3_order_param_nff_lower.xvg.\n")
	output_xvg = open(filename_xvg, 'w')
	output_xvg.write("@ title \"Evolution of lipid tails order parameters in lower leaflet\n")
	output_xvg.write("@ xaxis  label \"time (ns)\"\n")
	output_xvg.write("@ yaxis  label \"order parameter S2\"\n")
	output_xvg.write("@ autoscale ONREAD xaxes\n")
	output_xvg.write("@ TYPE XY\n")
	output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
	output_xvg.write("@ legend on\n")
	output_xvg.write("@ legend box on\n")
	output_xvg.write("@ legend loctype view\n")
	output_xvg.write("@ legend 0.98, 0.8\n")
	output_xvg.write("@ legend length " + str(2*len(lipids_handled["lower"])*3) + "\n")
	for s_index in range(0,len(lipids_handled["lower"])):
		output_xvg.write("@ s" + str(3*s_index) + " legend \"" + str(lipids_handled["lower"][s_index]) + " tail A (avg)\"\n")
		output_xvg.write("@ s" + str(3*s_index+1) + " legend \"" + str(lipids_handled["lower"][s_index]) + " tail B (avg)\"\n")
		output_xvg.write("@ s" + str(3*s_index+2) + " legend \"" + str(lipids_handled["lower"][s_index]) + " both (avg)\"\n")
		output_txt.write("1_3_order_param_nff_lower.xvg," + str((3*s_index)+1) +"," + str(lipids_handled["lower"][s_index]) + " tail A (avg)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["lower"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_lower.xvg," + str((3*s_index+1)+1) +"," + str(lipids_handled["lower"][s_index]) + " tail B (avg)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["lower"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_lower.xvg," + str((3*s_index+2)+1) +"," + str(lipids_handled["lower"][s_index]) + " both (avg)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["lower"][s_index])]) + "\n")
	for s_index in range(0,len(lipids_handled["lower"])):
		output_xvg.write("@ s" + str(3*len(lipids_handled["lower"])+3*s_index) + " legend \"" + str(lipids_handled["lower"][s_index]) + " tail A (std)\"\n")
		output_xvg.write("@ s" + str(3*len(lipids_handled["lower"])+3*s_index+1) + " legend \"" + str(lipids_handled["lower"][s_index]) + " tail (std)B\"\n")
		output_xvg.write("@ s" + str(3*len(lipids_handled["lower"])+3*s_index+2) + " legend \"" + str(lipids_handled["lower"][s_index]) + " both (std)\"\n")
		output_txt.write("1_3_order_param_nff_lower.xvg," + str(3*len(lipids_handled["lower"])+(3*s_index)+1) +"," + str(lipids_handled["lower"][s_index]) + " tail A (std)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["lower"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_lower.xvg," + str(3*len(lipids_handled["lower"])+(3*s_index+1)+1) +"," + str(lipids_handled["lower"][s_index]) + " tail B (std)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["lower"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_lower.xvg," + str(3*len(lipids_handled["lower"])+(3*s_index+2)+1) +"," + str(lipids_handled["lower"][s_index]) + " both (std)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["lower"][s_index])]) + "\n")
	output_txt.close()
	for frame in sorted(time_stamp.iterkeys()):
		results=str(time_stamp[frame])
		for s in lipids_handled["lower"]:
			results+="	" + str(round(op_tailA_avg_frame["lower"][s][frame],2)) + "	" + str(round(op_tailB_avg_frame["lower"][s][frame],2)) + "	" + str(round(op_both_avg_frame["lower"][s][frame],2))
		for s in lipids_handled["lower"]:
			results+="	" + str(round(op_tailA_std_frame["lower"][s][frame],2)) + "	" + str(round(op_tailB_std_frame["lower"][s][frame],2)) + "	" + str(round(op_both_std_frame["lower"][s][frame],2))
		output_xvg.write(results + "\n")
	output_xvg.close()

	return
def write_op_nff_xvg_smoothed():
	
	#lipids in upper leaflet
	#=======================
	filename_txt=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/smoothed/xvg/1_5_order_param_nff_upper_smoothed.txt'
	filename_xvg=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/smoothed/xvg/1_5_order_param_nff_upper_smoothed.xvg'
	output_txt = open(filename_txt, 'w')
	output_txt.write("@[lipid tail order parameters statistics - written by order_param v" + str(version_nb) + "]\n")
	output_txt.write("@Use this file as the argument of the -c option of the script 'xvg_animate' in order to make a time lapse movie of the data in 1_5_order_param_nff_upper_smoothed.xvg.\n")
	output_xvg = open(filename_xvg, 'w')
	output_xvg.write("@ title \"Evolution of lipid tails order parameters in upper leaflet\n")
	output_xvg.write("@ xaxis  label \"time (ns)\"\n")
	output_xvg.write("@ yaxis  label \"order parameter S2\"\n")
	output_xvg.write("@ autoscale ONREAD xaxes\n")
	output_xvg.write("@ TYPE XY\n")
	output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
	output_xvg.write("@ legend on\n")
	output_xvg.write("@ legend box on\n")
	output_xvg.write("@ legend loctype view\n")
	output_xvg.write("@ legend 0.98, 0.8\n")
	output_xvg.write("@ legend length " + str(2*len(lipids_handled["upper"])*3) + "\n")
	for s_index in range(0,len(lipids_handled["upper"])):
		output_xvg.write("@ s" + str(3*s_index) + " legend \"" + str(lipids_handled["upper"][s_index]) + " tail A (avg)\"\n")
		output_xvg.write("@ s" + str(3*s_index+1) + " legend \"" + str(lipids_handled["upper"][s_index]) + " tail B (avg)\"\n")
		output_xvg.write("@ s" + str(3*s_index+2) + " legend \"" + str(lipids_handled["upper"][s_index]) + " both (avg)\"\n")
		output_txt.write("1_3_order_param_nff_upper_smoothed.xvg," + str((3*s_index)+1) +"," + str(lipids_handled["upper"][s_index]) + " tail A (avg)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["upper"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_upper_smoothed.xvg," + str((3*s_index+1)+1) +"," + str(lipids_handled["upper"][s_index]) + " tail B (avg)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["upper"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_upper_smoothed.xvg," + str((3*s_index+2)+1) +"," + str(lipids_handled["upper"][s_index]) + " both (avg)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["upper"][s_index])]) + "\n")
	for s_index in range(0,len(lipids_handled["upper"])):
		output_xvg.write("@ s" + str(3*len(lipids_handled["upper"])+3*s_index) + " legend \"" + str(lipids_handled["upper"][s_index]) + " tail A (std)\"\n")
		output_xvg.write("@ s" + str(3*len(lipids_handled["upper"])+3*s_index+1) + " legend \"" + str(lipids_handled["upper"][s_index]) + " tail (std)B\"\n")
		output_xvg.write("@ s" + str(3*len(lipids_handled["upper"])+3*s_index+2) + " legend \"" + str(lipids_handled["upper"][s_index]) + " both (std)\"\n")
		output_txt.write("1_3_order_param_nff_upper_smoothed.xvg," + str(3*len(lipids_handled["upper"])+(3*s_index)+1) +"," + str(lipids_handled["upper"][s_index]) + " tail A (std)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["upper"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_upper_smoothed.xvg," + str(3*len(lipids_handled["upper"])+(3*s_index+1)+1) +"," + str(lipids_handled["upper"][s_index]) + " tail B (std)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["upper"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_upper_smoothed.xvg," + str(3*len(lipids_handled["upper"])+(3*s_index+2)+1) +"," + str(lipids_handled["upper"][s_index]) + " both (std)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["upper"][s_index])]) + "\n")
	output_txt.close()
	for t_index in range(0, numpy.size(time_smooth)):
		results=str(time_smooth[t_index])
		for s in lipids_handled["upper"]:
			results+="	" + str(round(op_tailA_avg_frame_smooth["upper"][s][t_index],2)) + "	" + str(round(op_tailB_avg_frame_smooth["upper"][s][t_index],2)) + "	" + str(round(op_both_avg_frame_smooth["upper"][s][t_index],2))
		for s in lipids_handled["upper"]:
			results+="	" + str(round(op_tailA_std_frame_smooth["upper"][s][t_index],2)) + "	" + str(round(op_tailB_std_frame_smooth["upper"][s][t_index],2)) + "	" + str(round(op_both_std_frame_smooth["upper"][s][t_index],2))
		output_xvg.write(results)
	output_xvg.close()

	#lipids in lower leaflet
	#=======================
	filename_txt=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/smoothed/xvg/1_5_order_param_nff_lower_smoothed.txt'
	filename_xvg=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/smoothed/xvg/1_5_order_param_nff_lower_smoothed.xvg'
	output_txt = open(filename_txt, 'w')
	output_txt.write("@[lipid tail order parameters statistics - written by order_param v" + str(version_nb) + "]\n")
	output_txt.write("@Use this file as the argument of the -c option of the script 'xvg_animate' in order to make a time lapse movie of the data in 1_5_order_param_nff_lower_smoothed.xvg.\n")
	output_xvg = open(filename_xvg, 'w')
	output_xvg.write("@ title \"Evolution of lipid tails order parameters in lower leaflet\n")
	output_xvg.write("@ xaxis  label \"time (ns)\"\n")
	output_xvg.write("@ yaxis  label \"order parameter S2\"\n")
	output_xvg.write("@ autoscale ONREAD xaxes\n")
	output_xvg.write("@ TYPE XY\n")
	output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
	output_xvg.write("@ legend on\n")
	output_xvg.write("@ legend box on\n")
	output_xvg.write("@ legend loctype view\n")
	output_xvg.write("@ legend 0.98, 0.8\n")
	output_xvg.write("@ legend length " + str(2*len(lipids_handled["lower"])*3) + "\n")
	for s_index in range(0,len(lipids_handled["lower"])):
		output_xvg.write("@ s" + str(3*s_index) + " legend \"" + str(lipids_handled["lower"][s_index]) + " tail A (avg)\"\n")
		output_xvg.write("@ s" + str(3*s_index+1) + " legend \"" + str(lipids_handled["lower"][s_index]) + " tail B (avg)\"\n")
		output_xvg.write("@ s" + str(3*s_index+2) + " legend \"" + str(lipids_handled["lower"][s_index]) + " both (avg)\"\n")
		output_txt.write("1_3_order_param_nff_lower_smoothed.xvg," + str((3*s_index)+1) +"," + str(lipids_handled["lower"][s_index]) + " tail A (avg)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["lower"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_lower_smoothed.xvg," + str((3*s_index+1)+1) +"," + str(lipids_handled["lower"][s_index]) + " tail B (avg)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["lower"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_lower_smoothed.xvg," + str((3*s_index+2)+1) +"," + str(lipids_handled["lower"][s_index]) + " both (avg)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["lower"][s_index])]) + "\n")
	for s_index in range(0,len(lipids_handled["lower"])):
		output_xvg.write("@ s" + str(3*len(lipids_handled["lower"])+3*s_index) + " legend \"" + str(lipids_handled["lower"][s_index]) + " tail A (std)\"\n")
		output_xvg.write("@ s" + str(3*len(lipids_handled["lower"])+3*s_index+1) + " legend \"" + str(lipids_handled["lower"][s_index]) + " tail (std)B\"\n")
		output_xvg.write("@ s" + str(3*len(lipids_handled["lower"])+3*s_index+2) + " legend \"" + str(lipids_handled["lower"][s_index]) + " both (std)\"\n")
		output_txt.write("1_3_order_param_nff_lower_smoothed.xvg," + str(3*len(lipids_handled["lower"])+(3*s_index)+1) +"," + str(lipids_handled["lower"][s_index]) + " tail A (std)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["lower"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_lower_smoothed.xvg," + str(3*len(lipids_handled["lower"])+(3*s_index+1)+1) +"," + str(lipids_handled["lower"][s_index]) + " tail B (std)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["lower"][s_index])]) + "\n")
		output_txt.write("1_3_order_param_nff_lower_smoothed.xvg," + str(3*len(lipids_handled["lower"])+(3*s_index+2)+1) +"," + str(lipids_handled["lower"][s_index]) + " both (std)," + mcolors.rgb2hex(colours_lipids[str(lipids_handled["lower"][s_index])]) + "\n")
	output_txt.close()
	for t_index in range(0, numpy.size(time_smooth)):
		results=str(time_smooth[t_index])
		for s in lipids_handled["lower"]:
			results+="	" + str(round(op_tailA_avg_frame_smooth["lower"][s][t_index],2)) + "	" + str(round(op_tailB_avg_frame_smooth["lower"][s][t_index],2)) + "	" + str(round(op_both_avg_frame_smooth["lower"][s][t_index],2))
		for s in lipids_handled["lower"]:
			results+="	" + str(round(op_tailA_std_frame_smooth["lower"][s][t_index],2)) + "	" + str(round(op_tailB_std_frame_smooth["lower"][s][t_index],2)) + "	" + str(round(op_both_std_frame_smooth["lower"][s][t_index],2))
		output_xvg.write(results + "\n")
	output_xvg.close()

	return
def graph_op_nff_xvg():
	
	#create filenames
	#----------------
	filename_png=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/png/1_2_order_param_nff.png'
	filename_svg=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/1_2_order_param_nff.svg'
	
	#create figure
	#-------------
	fig=plt.figure(figsize=(8, 6.2))
	fig.suptitle("Evolution of lipid tails order parameter")
	
	#create data
	#-----------
	tmp_time=[]
	tmp_op_both_avg_frame={}
	tmp_op_both_std_frame={}
	for l in ["lower","upper"]:
		tmp_op_both_avg_frame[l]={}
		tmp_op_both_std_frame[l]={}
		for s in lipids_handled[l]:
			tmp_op_both_avg_frame[l][s]=[]
			tmp_op_both_std_frame[l][s]=[]
	for frame in sorted(time_stamp.iterkeys()):
		tmp_time.append(time_stamp[frame])
		for l in ["lower","upper"]:
			for s in lipids_handled[l]:
				tmp_op_both_avg_frame[l][s].append(op_both_avg_frame[l][s][frame])
				tmp_op_both_std_frame[l][s].append(op_both_std_frame[l][s][frame])
				
	#plot data: upper leafet
	#-----------------------
	ax1 = fig.add_subplot(211)
	p_upper={}
	for s in lipids_handled["upper"]:
		p_upper[s]=plt.plot(tmp_time, tmp_op_both_avg_frame["upper"][s], color=colours_lipids[s], linewidth=3.0, label=str(s))
		p_upper[str(s + "_err")]=plt.fill_between(tmp_time, numpy.asarray(tmp_op_both_avg_frame["upper"][s])-numpy.asarray(tmp_op_both_std_frame["upper"][s]), numpy.asarray(tmp_op_both_avg_frame["upper"][s])+numpy.asarray(tmp_op_both_std_frame["upper"][s]), color=colours_lipids[s], alpha=0.2)
	fontP.set_size("small")
	ax1.legend(prop=fontP)
	plt.title("upper leaflet", fontsize="small")
	plt.xlabel('time (ns)', fontsize="small")
	plt.ylabel('order parameter', fontsize="small")
	
	#plot data: lower leafet
	#-----------------------
	ax2 = fig.add_subplot(212)
	p_lower={}
	for s in lipids_handled["lower"]:
		p_lower[s]=plt.plot(tmp_time, tmp_op_both_avg_frame["lower"][s], color=colours_lipids[s], linewidth=3.0, label=str(s))
		p_lower[str(s + "_err")]=plt.fill_between(tmp_time, numpy.asarray(tmp_op_both_avg_frame["lower"][s])-numpy.asarray(tmp_op_both_std_frame["lower"][s]), numpy.asarray(tmp_op_both_avg_frame["lower"][s])+numpy.asarray(tmp_op_both_std_frame["lower"][s]), color=colours_lipids[s], alpha=0.2)
	fontP.set_size("small")
	ax2.legend(prop=fontP)
	plt.title("lower leaflet", fontsize="small")
	plt.xlabel('time (ns)', fontsize="small")
	plt.ylabel('order parameter', fontsize="small")

	#save figure
	#-----------
	ax1.set_ylim(-0.5, 1)
	ax2.set_ylim(-0.5, 1)
	ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
	ax1.yaxis.set_major_locator(MaxNLocator(nbins=7))
	ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
	ax2.yaxis.set_major_locator(MaxNLocator(nbins=7))
	plt.setp(ax1.xaxis.get_majorticklabels(), fontsize="small" )
	plt.setp(ax1.yaxis.get_majorticklabels(), fontsize="small" )
	plt.setp(ax2.xaxis.get_majorticklabels(), fontsize="small" )
	plt.setp(ax2.yaxis.get_majorticklabels(), fontsize="small" )	
	plt.subplots_adjust(top=0.9, bottom=0.07, hspace=0.37, left=0.09, right=0.96)
	fig.savefig(filename_png)
	fig.savefig(filename_svg)
	plt.close()
	
	return
def graph_op_nff_xvg_smoothed():
	
	#create filenames
	#----------------
	filename_png=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/smoothed/png/1_4_order_param_nff_smoothed.png'
	filename_svg=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/smoothed/1_4_order_param_nff_smoothed.svg'
	
	#create figure
	#-------------
	fig=plt.figure(figsize=(8, 6.2))
	fig.suptitle("Evolution of lipid tails order parameter")
					
	#plot data: upper leafet
	#-----------------------
	ax1 = fig.add_subplot(211)
	p_upper={}
	for s in lipids_handled["upper"]:
		p_upper[s]=plt.plot(time_smooth, op_both_avg_frame_smooth["upper"][s], color=colours_lipids[s], linewidth=3.0, label=str(s))
		p_upper[str(s + "_err")]=plt.fill_between(time_smooth, numpy.asarray(op_both_avg_frame_smooth["upper"][s])-numpy.asarray(op_both_std_frame_smooth["upper"][s]), numpy.asarray(op_both_avg_frame_smooth["upper"][s])+numpy.asarray(op_both_std_frame_smooth["upper"][s]), color=colours_lipids[s], alpha=0.2)
	fontP.set_size("small")
	ax1.legend(prop=fontP)
	plt.title("upper leaflet", fontsize="small")
	plt.xlabel('time (ns)', fontsize="small")
	plt.ylabel('order parameter', fontsize="small")
	
	#plot data: lower leafet
	#-----------------------
	ax2 = fig.add_subplot(212)
	p_lower={}
	for s in lipids_handled["lower"]:
		p_lower[s]=plt.plot(time_smooth, op_both_avg_frame_smooth["lower"][s], color=colours_lipids[s], linewidth=3.0, label=str(s))
		p_lower[str(s + "_err")]=plt.fill_between(time_smooth, numpy.asarray(op_both_avg_frame_smooth["lower"][s])-numpy.asarray(op_both_std_frame_smooth["lower"][s]), numpy.asarray(op_both_avg_frame_smooth["lower"][s])+numpy.asarray(op_both_std_frame_smooth["lower"][s]), color=colours_lipids[s], alpha=0.2)
	fontP.set_size("small")
	ax2.legend(prop=fontP)
	plt.title("lower leaflet", fontsize="small")
	plt.xlabel('time (ns)', fontsize="small")
	plt.ylabel('order parameter', fontsize="small")

	#save figure
	#-----------
	ax1.set_ylim(-0.5, 1)
	ax2.set_ylim(-0.5, 1)
	ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
	ax1.yaxis.set_major_locator(MaxNLocator(nbins=7))
	ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
	ax2.yaxis.set_major_locator(MaxNLocator(nbins=7))
	plt.setp(ax1.xaxis.get_majorticklabels(), fontsize="small" )
	plt.setp(ax1.yaxis.get_majorticklabels(), fontsize="small" )
	plt.setp(ax2.xaxis.get_majorticklabels(), fontsize="small" )
	plt.setp(ax2.yaxis.get_majorticklabels(), fontsize="small" )	
	plt.subplots_adjust(top=0.9, bottom=0.07, hspace=0.37, left=0.09, right=0.96)
	fig.savefig(filename_png)
	fig.savefig(filename_svg)
	plt.close()
	
	return

#flipflopping lipids
def write_op_ff_xvg():
	
	#upper to lower flipflops
	#========================
	if numpy.size(lipids_ff_u2l_index)>0:
		filename_txt=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/xvg/4_3_order_param_ff_u2l.txt'
		filename_xvg=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/xvg/4_3_order_param_ff_u2l.xvg'
		output_txt = open(filename_txt, 'w')
		output_txt.write("@[lipid tail order parameters statistics - written by order_param v" + str(version_nb) + "]\n")
		output_txt.write("@Use this file as the argument of the -c option of the script 'xvg_animate' in order to make a time lapse movie of the data in 4_3_order_param_ff_u2l.xvg.\n")
		output_xvg = open(filename_xvg, 'w')
		output_xvg.write("@ title \"Evolution of the tail order parameters of flipflopping lipids\"\n")
		output_xvg.write("@ xaxis  label \"time (ns)\"\n")
		output_xvg.write("@ yaxis  label \"order parameter S2\"\n")
		output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
		output_xvg.write("@ legend on\n")
		output_xvg.write("@ legend box on\n")
		output_xvg.write("@ legend loctype view\n")
		output_xvg.write("@ legend 0.98, 0.8\n")
		output_xvg.write("@ legend length " + str(lipids_ff_nb*3) + "\n")
		for l_index in range(0,len(lipids_ff_u2l_index)):
			l=lipids_ff_u2l_index[l_index]
			output_xvg.write("@ s" + str(3*l_index) + " legend \"" + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail A\"\n")
			output_xvg.write("@ s" + str(3*l_index+1) + " legend \"" + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail B\"\n")
			output_xvg.write("@ s" + str(3*l_index+2) + " legend \"" + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " both\"\n")
			output_txt.write("4_3_order_param_ff_u2l.xvg," + str((3*l_index)+1) + "," + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail A,auto\n")
			output_txt.write("4_3_order_param_ff_u2l.xvg," + str((3*l_index+1)+1) + "," + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail B,auto\n")
			output_txt.write("4_3_order_param_ff_u2l.xvg," + str((3*l_index+2)+1) +"," + "," + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " both,auto\n")
		output_txt.close()
		for frame in sorted(time_stamp.iterkeys()):
			results=str(time_stamp[frame])
			for l in lipids_ff_u2l_index:
				results+="	" + str(round(op_ff_tailA[l][frame],2)) + "	" + str(round(op_ff_tailB[l][frame],2)) + "	" + str(round(op_ff_both[l][frame],2))
			output_xvg.write(results + "\n")
		output_xvg.close()
	
	#lower to upper flipflops
	#========================
	if numpy.size(lipids_ff_l2u_index)>0:
		filename_txt=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/xvg/4_3_order_param_ff_l2u.txt'
		filename_xvg=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/xvg/4_3_order_param_ff_l2u.xvg'
		output_txt = open(filename_txt, 'w')
		output_txt.write("@[lipid tail order parameters statistics - written by order_param v" + str(version_nb) + "]\n")
		output_txt.write("@Use this file as the argument of the -c option of the script 'xvg_animate' in order to make a time lapse movie of the data in 4_3_order_param_ff_l2u.xvg.\n")
		output_xvg = open(filename_xvg, 'w')
		output_xvg.write("@ title \"Evolution of the tail order parameters of flipflopping lipids\"\n")
		output_xvg.write("@ xaxis  label \"time (ns)\"\n")
		output_xvg.write("@ yaxis  label \"order parameter S2\"\n")
		output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
		output_xvg.write("@ legend on\n")
		output_xvg.write("@ legend box on\n")
		output_xvg.write("@ legend loctype view\n")
		output_xvg.write("@ legend 0.98, 0.8\n")
		output_xvg.write("@ legend length " + str(lipids_ff_nb*3) + "\n")
		for l_index in range(0,len(lipids_ff_l2u_index)):
			l=lipids_ff_l2u_index[l_index]
			output_xvg.write("@ s" + str(3*l_index) + " legend \"" + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail A\"\n")
			output_xvg.write("@ s" + str(3*l_index+1) + " legend \"" + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail B\"\n")
			output_xvg.write("@ s" + str(3*l_index+2) + " legend \"" + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " both\"\n")
			output_txt.write("4_3_order_param_ff_l2u.xvg," + str((3*l_index)+1) + "," + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail A,auto\n")
			output_txt.write("4_3_order_param_ff_l2u.xvg," + str((3*l_index+1)+1) + "," + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail B,auto\n")
			output_txt.write("4_3_order_param_ff_l2u.xvg," + str((3*l_index+2)+1) +"," + "," + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " both,auto\n")
		output_txt.close()
		for frame in sorted(time_stamp.iterkeys()):
			results=str(time_stamp[frame])
			for l in lipids_ff_l2u_index:
				results+="	" + str(round(op_ff_tailA[l][frame],2)) + "	" + str(round(op_ff_tailB[l][frame],2)) + "	" + str(round(op_ff_both[l][frame],2))
			output_xvg.write(results + "\n")
		output_xvg.close()

	return
def write_op_ff_xvg_smoothed():

	#upper to lower flipflops
	#========================
	if numpy.size(lipids_ff_u2l_index)>0:
		filename_txt=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/smoothed/xvg/4_5_order_param_ff_u2l_smoothed.txt'
		filename_xvg=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/smoothed/xvg/4_5_order_param_ff_u2l_smoothed.xvg'
		output_txt = open(filename_txt, 'w')
		output_txt.write("@[lipid tail order parameters statistics - written by order_param v" + str(version_nb) + "]\n")
		output_txt.write("@Use this file as the argument of the -c option of the script 'xvg_animate' in order to make a time lapse movie of the data in 4_5_order_param_ff_u2l_smoothed.xvg.\n")
		output_xvg = open(filename_xvg, 'w')
		output_xvg.write("@ title \"Evolution of the tail order parameters of flipflopping lipids\"\n")
		output_xvg.write("@ xaxis  label \"time (ns)\"\n")
		output_xvg.write("@ yaxis  label \"order parameter S2\"\n")
		output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
		output_xvg.write("@ legend on\n")
		output_xvg.write("@ legend box on\n")
		output_xvg.write("@ legend loctype view\n")
		output_xvg.write("@ legend 0.98, 0.8\n")
		output_xvg.write("@ legend length " + str(lipids_ff_nb*3) + "\n")
		for l_index in range(0,len(lipids_ff_u2l_index)):
			l=lipids_ff_u2l_index[l_index]
			output_xvg.write("@ s" + str(3*l_index) + " legend \"" + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail A\"\n")
			output_xvg.write("@ s" + str(3*l_index+1) + " legend \"" + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail B\"\n")
			output_xvg.write("@ s" + str(3*l_index+2) + " legend \"" + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " both\"\n")
			output_txt.write("4_3_order_param_ff_u2l_smoothed.xvg," + str((3*l_index)+1) + "," + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail A,auto\n")
			output_txt.write("4_3_order_param_ff_u2l_smoothed.xvg," + str((3*l_index+1)+1) + "," + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail B,auto\n")
			output_txt.write("4_3_order_param_ff_u2l_smoothed.xvg," + str((3*l_index+2)+1) +"," + "," + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " both,auto\n")
		output_txt.close()
		for t_index in range(0, numpy.size(time_smooth)):
			results=str(time_smooth[t_index])
			for l in lipids_ff_u2l_index:
				results+="	" + str(round(op_ff_tailA_smooth[l][t_index],2)) + "	" + str(round(op_ff_tailB_smooth[l][t_index],2)) + "	" + str(round(op_ff_both_smooth[l][t_index],2))
			output_xvg.write(results + "\n")
		output_xvg.close()
	
	#lower to upper flipflops
	#========================
	if numpy.size(lipids_ff_l2u_index)>0:
		filename_txt=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/smoothed/xvg/4_5_order_param_ff_l2u_smoothed.txt'
		filename_xvg=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/smoothed/xvg/4_5_order_param_ff_l2u_smoothed.xvg'
		output_txt = open(filename_txt, 'w')
		output_txt.write("@[lipid tail order parameters statistics - written by order_param v" + str(version_nb) + "]\n")
		output_txt.write("@Use this file as the argument of the -c option of the script 'xvg_animate' in order to make a time lapse movie of the data in 4_5_order_param_ff_l2u_smoothed.xvg.\n")
		output_xvg = open(filename_xvg, 'w')
		output_xvg.write("@ title \"Evolution of the tail order parameters of flipflopping lipids\"\n")
		output_xvg.write("@ xaxis  label \"time (ns)\"\n")
		output_xvg.write("@ yaxis  label \"order parameter S2\"\n")
		output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
		output_xvg.write("@ legend on\n")
		output_xvg.write("@ legend box on\n")
		output_xvg.write("@ legend loctype view\n")
		output_xvg.write("@ legend 0.98, 0.8\n")
		output_xvg.write("@ legend length " + str(lipids_ff_nb*3) + "\n")
		for l_index in range(0,len(lipids_ff_l2u_index)):
			l=lipids_ff_l2u_index[l_index]
			output_xvg.write("@ s" + str(3*l_index) + " legend \"" + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail A\"\n")
			output_xvg.write("@ s" + str(3*l_index+1) + " legend \"" + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail B\"\n")
			output_xvg.write("@ s" + str(3*l_index+2) + " legend \"" + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " both\"\n")
			output_txt.write("4_3_order_param_ff_l2u_smoothed.xvg," + str((3*l_index)+1) + "," + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail A,auto\n")
			output_txt.write("4_3_order_param_ff_l2u_smoothed.xvg," + str((3*l_index+1)+1) + "," + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " tail B,auto\n")
			output_txt.write("4_3_order_param_ff_l2u_smoothed.xvg," + str((3*l_index+2)+1) +"," + "," + str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]) + " both,auto\n")
		output_txt.close()
		for t_index in range(0, numpy.size(time_smooth)):
			results=str(time_smooth[t_index])
			for l in lipids_ff_l2u_index:
				results+="	" + str(round(op_ff_tailA_smooth[l][t_index],2)) + "	" + str(round(op_ff_tailB_smooth[l][t_index],2)) + "	" + str(round(op_ff_both_smooth[l][t_index],2))
			output_xvg.write(results + "\n")
		output_xvg.close()

	return
def graph_op_ff_xvg():

	#upper to lower flipflops
	#========================
	if numpy.size(lipids_ff_u2l_index)>0:

		#create filenames
		#----------------
		filename_png=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/png/4_2_order_param_ff_u2l.png'
		filename_svg=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/4_2_order_param_ff_u2l.svg'
		
		#create figure
		#-------------
		fig=plt.figure(figsize=(8, 6.2))
		fig.suptitle("Flipflopping lipids: upper to lower")
	
		#create data
		#-----------
		tmp_time=[]
		tmp_z_ff={}
		tmp_z_upper=[]
		tmp_z_lower=[]
		tmp_op_ff_both={}
		for l in lipids_ff_u2l_index:
			tmp_z_ff[l]=[]
			tmp_op_ff_both[l]=[]
		for frame in sorted(time_stamp.iterkeys()):
			tmp_time.append(time_stamp[frame])
			tmp_z_upper.append(z_upper[frame])
			tmp_z_lower.append(z_lower[frame])
			for l in lipids_ff_u2l_index:
				tmp_z_ff[l].append(z_ff[l][frame])
				tmp_op_ff_both[l].append(op_ff_both[l][frame])
	
		#plot data: order paramter
		#-------------------------
		ax1 = fig.add_subplot(211)
		p_upper={}
		for l in lipids_ff_u2l_index:
			p_upper[l]=plt.plot(tmp_time, tmp_op_ff_both[l], label=str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]))
		fontP.set_size("small")
		ax1.legend(prop=fontP)
		plt.xlabel('time (ns)', fontsize="small")
		plt.ylabel('order parameter', fontsize="small")
	
		#plot data: z coordinate
		#-----------------------
		ax2 = fig.add_subplot(212)
		p_lower={}
		p_lower["upper"]=plt.plot(tmp_time, tmp_z_upper, linestyle='dashed', color='k')
		p_lower["lower"]=plt.plot(tmp_time, tmp_z_lower, linestyle='dashed', color='k')
		for l in lipids_ff_u2l_index:
			p_lower[l]=plt.plot(tmp_time, tmp_z_ff[l], label=str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]))
		fontP.set_size("small")
		ax2.legend(prop=fontP)
		plt.xlabel('time (ns)', fontsize="small")
		plt.ylabel('z coordinate', fontsize="small")
		
		#save figure
		#-----------
		ax1.set_ylim(-0.5, 1)
		ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax1.yaxis.set_major_locator(MaxNLocator(nbins=7))
		ax2.set_ylim(-40, 40)
		ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
		plt.setp(ax1.xaxis.get_majorticklabels(), fontsize="small" )
		plt.setp(ax1.yaxis.get_majorticklabels(), fontsize="small" )
		plt.subplots_adjust(top=0.9, bottom=0.07, hspace=0.37, left=0.09, right=0.96)
		fig.savefig(filename_png)
		fig.savefig(filename_svg)
		plt.close()

	#lower to upper flipflops
	#========================
	if numpy.size(lipids_ff_l2u_index)>0:

		#create filenames
		#----------------
		filename_png=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/png/4_2_order_param_ff_l2u.png'
		filename_svg=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/4_2_order_param_ff_l2u.svg'
		
		#create figure
		#-------------
		fig=plt.figure(figsize=(8, 6.2))
		fig.suptitle("Flipflopping lipids: upper to lower")
	
		#create data
		#-----------
		tmp_time=[]
		tmp_z_ff={}
		tmp_z_upper=[]
		tmp_z_lower=[]
		tmp_op_ff_both={}
		for l in lipids_ff_l2u_index:
			tmp_z_ff[l]=[]
			tmp_op_ff_both[l]=[]
		for frame in sorted(time_stamp.iterkeys()):
			tmp_time.append(time_stamp[frame])
			tmp_z_upper.append(z_upper[frame])
			tmp_z_lower.append(z_lower[frame])
			for l in lipids_ff_l2u_index:
				tmp_z_ff[l].append(z_ff[l][frame])
				tmp_op_ff_both[l].append(op_ff_both[l][frame])
	
		#plot data: order paramter
		#-------------------------
		ax1 = fig.add_subplot(211)
		p_upper={}
		for l in lipids_ff_l2u_index:
			p_upper[l]=plt.plot(tmp_time, tmp_op_ff_both[l], label=str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]))
		fontP.set_size("small")
		ax1.legend(prop=fontP)
		plt.xlabel('time (ns)', fontsize="small")
		plt.ylabel('order parameter', fontsize="small")
	
		#plot data: z coordinate
		#-----------------------
		ax2 = fig.add_subplot(212)
		p_lower={}
		p_lower["upper"]=plt.plot(tmp_time, tmp_z_upper, linestyle='dashed', color='k')
		p_lower["lower"]=plt.plot(tmp_time, tmp_z_lower, linestyle='dashed', color='k')
		for l in lipids_ff_l2u_index:
			p_lower[l]=plt.plot(tmp_time, tmp_z_ff[l], label=str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]))
		fontP.set_size("small")
		ax1.legend(prop=fontP)
		plt.xlabel('time (ns)', fontsize="small")
		plt.ylabel('z coordinate', fontsize="small")
		
		#save figure
		#-----------
		ax1.set_ylim(-0.5, 1)
		ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax1.yaxis.set_major_locator(MaxNLocator(nbins=7))
		ax2.set_ylim(-40, 40)
		ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
		plt.setp(ax1.xaxis.get_majorticklabels(), fontsize="small" )
		plt.setp(ax1.yaxis.get_majorticklabels(), fontsize="small" )
		plt.subplots_adjust(top=0.9, bottom=0.07, hspace=0.37, left=0.09, right=0.96)
		fig.savefig(filename_png)
		fig.savefig(filename_svg)
		plt.close()

	return
def graph_op_ff_xvg_smoothed():
	
	#upper to lower flipflops
	#========================
	if numpy.size(lipids_ff_u2l_index)>0:

		#create filenames
		#----------------
		filename_png=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/smoothed/png/4_4_order_param_ff_u2l_smoothed.png'
		filename_svg=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/smoothed/4_4_order_param_ff_u2l_smoothed.svg'
		
		#create figure
		#-------------
		fig=plt.figure(figsize=(8, 6.2))
		fig.suptitle("Flipflopping lipids: upper to lower")
	
		#create data
		#-----------
		tmp_time=[]
		tmp_z_ff={}
		tmp_z_upper=[]
		tmp_z_lower=[]
		tmp_op_ff_both={}
		for l in lipids_ff_u2l_index:
			tmp_z_ff[l]=[]
			tmp_op_ff_both[l]=[]
		for t_index in range(0, numpy.size(time_smooth)):
			tmp_time.append(time_smooth[t_index])
			tmp_z_upper.append(z_upper_smooth[t_index])
			tmp_z_lower.append(z_lower_smooth[t_index])
			for l in lipids_ff_u2l_index:
				tmp_z_ff[l].append(z_ff_smooth[l][t_index])
				tmp_op_ff_both[l].append(op_ff_both_smooth[l][t_index])
		
		#plot data: order paramter
		#-------------------------
		ax1 = fig.add_subplot(211)
		p_upper={}
		for l in lipids_ff_u2l_index:
			p_upper[l]=plt.plot(tmp_time, tmp_op_ff_both[l], label=str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]))
		fontP.set_size("small")
		ax1.legend(prop=fontP)
		plt.xlabel('time (ns)', fontsize="small")
		plt.ylabel('order parameter', fontsize="small")
	
		#plot data: z coordinate
		#-----------------------
		ax2 = fig.add_subplot(212)
		p_lower={}
		p_lower["upper"]=plt.plot(tmp_time, tmp_z_upper, linestyle='dashed', color='k')
		p_lower["lower"]=plt.plot(tmp_time, tmp_z_lower, linestyle='dashed', color='k')
		for l in lipids_ff_u2l_index:
			p_lower[l]=plt.plot(tmp_time, tmp_z_ff[l], label=str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]))
		fontP.set_size("small")
		ax2.legend(prop=fontP)
		plt.xlabel('time (ns)', fontsize="small")
		plt.ylabel('z coordinate', fontsize="small")
		
		#save figure
		#-----------
		ax1.set_ylim(-0.5, 1)
		ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax1.yaxis.set_major_locator(MaxNLocator(nbins=7))
		ax2.set_ylim(-40, 40)
		ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
		plt.setp(ax1.xaxis.get_majorticklabels(), fontsize="small" )
		plt.setp(ax1.yaxis.get_majorticklabels(), fontsize="small" )
		plt.subplots_adjust(top=0.9, bottom=0.07, hspace=0.37, left=0.09, right=0.96)
		fig.savefig(filename_png)
		fig.savefig(filename_svg)
		plt.close()

	#lower to upper flipflops
	#========================
	if numpy.size(lipids_ff_l2u_index)>0:

		#create filenames
		#----------------
		filename_png=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/smoothed/png/4_4_order_param_ff_l2u_smoothed.png'
		filename_svg=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/smoothed/4_4_order_param_ff_l2u_smoothed.svg'
		
		#create figure
		#-------------
		fig=plt.figure(figsize=(8, 6.2))
		fig.suptitle("Flipflopping lipids: lower to lower")
		
		#create data
		#-----------
		tmp_time=[]
		tmp_z_ff={}
		tmp_z_upper=[]
		tmp_z_lower=[]
		tmp_op_ff_both={}
		for l in lipids_ff_l2u_index:
			tmp_z_ff[l]=[]
			tmp_op_ff_both[l]=[]
		for t_index in range(0, numpy.size(time_smooth)):
			tmp_time.append(time_smooth[t_index])
			tmp_z_upper.append(z_upper_smooth[t_index])
			tmp_z_lower.append(z_lower_smooth[t_index])
			for l in lipids_ff_l2u_index:
				tmp_z_ff[l].append(z_ff_smooth[l][t_index])
				tmp_op_ff_both[l].append(op_ff_both_smooth[l][t_index])

		#plot data: order paramter
		#-------------------------
		ax1 = fig.add_subplot(211)
		p_upper={}
		for l in lipids_ff_l2u_index:
			p_upper[l]=plt.plot(tmp_time, tmp_op_ff_both[l], label=str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]))
		fontP.set_size("small")
		ax1.legend(prop=fontP)
		plt.xlabel('time (ns)', fontsize="small")
		plt.ylabel('order parameter', fontsize="small")
	
		#plot data: z coordinate
		#-----------------------
		ax2 = fig.add_subplot(212)
		p_lower={}
		p_lower["upper"]=plt.plot(tmp_time, tmp_z_upper, linestyle='dashed', color='k')
		p_lower["lower"]=plt.plot(tmp_time, tmp_z_lower, linestyle='dashed', color='k')
		for l in lipids_ff_l2u_index:
			p_lower[l]=plt.plot(tmp_time, tmp_z_ff[l], label=str(lipids_ff_info[l][0]) + " " + str(lipids_ff_info[l][1]))
		fontP.set_size("small")
		ax1.legend(prop=fontP)
		plt.xlabel('time (ns)', fontsize="small")
		plt.ylabel('z coordinate', fontsize="small")
		
		#save figure
		#-----------
		ax1.set_ylim(-0.5, 1)
		ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax1.yaxis.set_major_locator(MaxNLocator(nbins=7))
		ax2.set_ylim(-40, 40)
		ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
		plt.setp(ax1.xaxis.get_majorticklabels(), fontsize="small" )
		plt.setp(ax1.yaxis.get_majorticklabels(), fontsize="small" )
		plt.subplots_adjust(top=0.9, bottom=0.07, hspace=0.37, left=0.09, right=0.96)
		fig.savefig(filename_png)
		fig.savefig(filename_svg)
		plt.close()

	return

#annotations
#===========
def write_frame_stat(f_nb, f_index, t):

	#case: gro file or xtc summary
	#=============================
	if f_index=="all" and t=="all":
		#nff lipids
		#----------
		#create file
		filename_details=os.getcwd() + '/' + str(args.output_folder) + '/1_nff/1_1_order_param_nff.stat'
		output_stat = open(filename_details, 'w')		
		output_stat.write("[lipid tail order parameters statistics - written by order_param v" + str(version_nb) + "]\n")
		output_stat.write("\n")
	
		#general info
		output_stat.write("1. membrane composition\n")
		output_stat.write(membrane_comp["upper"] + "\n")
		output_stat.write(membrane_comp["lower"] + "\n")
		tmp_string=str(lipids_handled["both"][0])
		for s in lipids_handled["both"][1:]:
			tmp_string+=", " + str(s)
		output_stat.write("2. lipid species processed: " + str(tmp_string) + "\n")
		if args.xtcfilename!="no":
			output_stat.write("3. nb frames processed:	" + str(nb_frames_processed) + " (" + str(nb_frames_xtc) + " frames in xtc, step=" + str(args.frames_dt) + ")\n")
		output_stat.write("\n")
		output_stat.write("lipid orientation with bilayer normal\n")
		output_stat.write(" P2=1    : parallel\n")
		output_stat.write(" P2=0    : random\n")
		output_stat.write(" P2=-0.5 : orthogonal\n")
	
		#average
		output_stat.write("\n")
		output_stat.write("average order parameter: " + str(round((op_both_avg["upper"]["all"]*lipids_nff_sele_nb["upper"]["all"]+op_both_avg["lower"]["all"]*lipids_nff_sele_nb["lower"]["all"])/float(lipids_nff_sele_nb["upper"]["all"]+lipids_nff_sele_nb["lower"]["all"]),2)) + "\n")
	
		#lipids in upper leaflet
		output_stat.write("\n")
		output_stat.write("upper leaflet\n")
		output_stat.write("=============\n")
		output_stat.write("avg	nb	tail A	tail B	 both\n")
		output_stat.write("-------------------------------------\n")
		for s in lipids_handled["upper"]:
			output_stat.write(str(s) + "	" + str(lipids_nff_sele["upper"][s].numberOfResidues()) + "	" + str(round(op_tailA_avg["upper"][s],2)) + "	" + str(round(op_tailB_avg["upper"][s],2)) + "	 " + str(round(op_both_avg["upper"][s],2)) + "\n")
		output_stat.write("-------------------------------------\n")
		output_stat.write("all	" + str(lipids_nff_sele_nb["upper"]["all"]) + "	" + str(round(op_tailA_avg["upper"]["all"],2)) + "	" + str(round(op_tailB_avg["upper"]["all"],2)) + "	 " + str(round(op_both_avg["upper"]["all"],2)) + "\n")		
		output_stat.write("\n")
		output_stat.write("std	nb	tail A	tail B	 both\n")
		output_stat.write("-------------------------------------\n")
		for s in lipids_handled["upper"]:
			output_stat.write(str(s) + "	" + str(lipids_nff_sele["upper"][s].numberOfResidues()) + "	" + str(round(op_tailA_std["upper"][s],2)) + "	" + str(round(op_tailB_std["upper"][s],2)) + "	 " + str(round(op_both_std["upper"][s],2)) + "\n")
		output_stat.write("-------------------------------------\n")
		output_stat.write("all	" + str(lipids_nff_sele_nb["upper"]["all"]) + "	" + str(round(op_tailA_std["upper"]["all"],2)) + "	" + str(round(op_tailB_std["upper"]["all"],2)) + "	 " + str(round(op_both_std["upper"]["all"],2)) + "\n")		
	
		#lipids in lower leaflet
		output_stat.write("\n")
		output_stat.write("lower leaflet\n")
		output_stat.write("=============\n")
		output_stat.write("avg	nb	tail A	tail B	 both\n")
		output_stat.write("-------------------------------------\n")
		for s in lipids_handled["lower"]:
			output_stat.write(str(s) + "	" + str(lipids_nff_sele["lower"][s].numberOfResidues()) + "	" + str(round(op_tailA_avg["lower"][s],2)) + "	" + str(round(op_tailB_avg["lower"][s],2)) + "	 " + str(round(op_both_avg["lower"][s],2)) + "\n")
		output_stat.write("-------------------------------------\n")
		output_stat.write("all	" + str(lipids_nff_sele_nb["lower"]["all"]) + "	" + str(round(op_tailA_avg["lower"]["all"],2)) + "	" + str(round(op_tailB_avg["lower"]["all"],2)) + "	 " + str(round(op_both_avg["lower"]["all"],2)) + "\n")		
		output_stat.write("\n")
		output_stat.write("std	nb	tail A	tail B	 both\n")
		output_stat.write("-------------------------------------\n")
		for s in lipids_handled["lower"]:
			output_stat.write(str(s) + "	" + str(lipids_nff_sele["lower"][s].numberOfResidues()) + "	" + str(round(op_tailA_std["lower"][s],2)) + "	" + str(round(op_tailB_std["lower"][s],2)) + "	 " + str(round(op_both_std["lower"][s],2)) + "\n")
		output_stat.write("-------------------------------------\n")
		output_stat.write("all	" + str(lipids_nff_sele_nb["lower"]["all"]) + "	" + str(round(op_tailA_std["lower"]["all"],2)) + "	" + str(round(op_tailB_std["lower"]["all"],2)) + "	 " + str(round(op_both_std["lower"]["all"],2)) + "\n")		

		output_stat.close()
	
		#ff lipids
		#---------
		if args.selection_file_ff!="no":
			filename_details=os.getcwd() + '/' + str(args.output_folder) + '/4_ff/4_1_order_param_ff.stat'
			output_stat = open(filename_details, 'w')		
			output_stat.write("[lipid tail order parameters statistics - written by order_param v" + str(version_nb) + "]\n")
			output_stat.write("\n")
		
			#general info
			output_stat.write("1. membrane composition\n")
			output_stat.write(membrane_comp["upper"] + "\n")
			output_stat.write(membrane_comp["lower"] + "\n")
			tmp_string=str(lipids_handled["both"][0])
			for s in lipids_handled["both"][1:]:
				tmp_string+=", " + str(s)
			output_stat.write("2. lipid species processed: " + str(tmp_string) + "\n")
			if args.xtcfilename!="no":
				output_stat.write("3. nb frames processed:	" + str(nb_frames_processed) + " (" + str(nb_frames_xtc) + " frames in xtc, step=" + str(args.frames_dt) + ")\n")
			output_stat.write("\n")
			output_stat.write("lipid orientation with bilayer normal\n")
			output_stat.write(" P2=1    : parallel\n")
			output_stat.write(" P2=0    : random\n")
			output_stat.write(" P2=-0.5 : orthogonal\n")
		
			#upper to lower
			if numpy.size(lipids_ff_u2l_index)>0:
				output_stat.write("\n")
				output_stat.write("upper to lower\n")
				output_stat.write("==============\n")
				output_stat.write("specie	resid	tail A	tail B  both\n")
				output_stat.write("-------------------------------------\n")
				for l in lipids_ff_u2l_index:
					output_stat.write(str(lipids_ff_info[l][0]) + "	" + str(lipids_ff_info[l][1]) + "	" + str(round(op_ff_tailA[l][1],2)) + "	" + str(round(op_ff_tailB[l][1],2)) + "	" + str(round(op_ff_both[l][1],2)) + "\n")
			
			#lower to upper
			if numpy.size(lipids_ff_l2u_index)>0:
				output_stat.write("\n")
				output_stat.write("lower to upper\n")
				output_stat.write("==============\n")
				output_stat.write("specie	resid	tail A	tail B  both\n")
				output_stat.write("-------------------------------------\n")
				for l in lipids_ff_l2u_index:
					output_stat.write(str(lipids_ff_info[l][0]) + "	" + str(lipids_ff_info[l][1]) + "	" + str(round(op_ff_tailA[l][1],2)) + "	" + str(round(op_ff_tailB[l][1],2)) + "	" + str(round(op_ff_both[l][1],2)) + "\n")
			output_stat.close()

	#case: xtc snapshot
	#==================
	else:
		#nff lipids
		#----------
		#create file
		filename_details=os.getcwd() + '/' + str(args.output_folder) + '/2_snapshots/' + args.xtcfilename[:-4] + '_annotated_orderparam_' + str(int(t)).zfill(5) + 'ns_nff.stat'
		output_stat = open(filename_details, 'w')		
		output_stat.write("[lipid tail order parameters statistics - written by order_param v" + str(version_nb) + "]\n")
		output_stat.write("\n")
	
		#general info
		output_stat.write("1. membrane composition\n")
		output_stat.write(membrane_comp["upper"] + "\n")
		output_stat.write(membrane_comp["lower"] + "\n")
		tmp_string=str(lipids_handled["both"][0])
		for s in lipids_handled["both"][1:]:
			tmp_string+=", " + str(s)
		output_stat.write("\n")
		output_stat.write("2. lipid species processed: " + str(tmp_string) + "\n")
		output_stat.write("\n")
		output_stat.write("3. nb frames processed:	" + str(nb_frames_processed) + " (" + str(nb_frames_xtc) + " frames in xtc, step=" + str(args.frames_dt) + ")\n")
		output_stat.write("\n")
		output_stat.write("4. time: " + str(t) + "ns (frame " + str(f_nb) + "/" + str(nb_frames_xtc) + ")\n")
		output_stat.write("\n")
		output_stat.write("lipid orientation with bilayer normal\n")
		output_stat.write(" P2=1    : parallel\n")
		output_stat.write(" P2=0    : random\n")
		output_stat.write(" P2=-0.5 : orthogonal\n")
			
		#lipids in upper leaflet
		output_stat.write("\n")
		output_stat.write("upper leaflet\n")
		output_stat.write("=============\n")
		output_stat.write("avg	nb	tail A	tail B	 both\n")
		output_stat.write("-------------------------------------\n")
		for s in lipids_handled["upper"]:
			output_stat.write(str(s) + "	" + str(lipids_nff_sele["upper"][s].numberOfResidues()) + "	" + str(round(op_tailA_avg_frame["upper"][s][f_nb],2)) + "	" + str(round(op_tailA_avg_frame["upper"][s][f_nb],2)) + "	 " + str(round(op_both_avg_frame["upper"][s][f_nb],2)) + "\n")
		output_stat.write("\n")
		output_stat.write("std	nb	tail A	tail B	 both\n")
		output_stat.write("-------------------------------------\n")
		for s in lipids_handled["upper"]:
			output_stat.write(str(s) + "	" + str(lipids_nff_sele["upper"][s].numberOfResidues()) + "	" + str(round(op_tailA_std_frame["upper"][s][f_nb],2)) + "	" + str(round(op_tailA_std_frame["upper"][s][f_nb],2)) + "	 " + str(round(op_both_std_frame["upper"][s][f_nb],2)) + "\n")
	
		#lipids in lower leaflet
		output_stat.write("\n")
		output_stat.write("lower leaflet\n")
		output_stat.write("=============\n")
		output_stat.write("avg	nb	tail A	tail B	 both\n")
		output_stat.write("-------------------------------------\n")
		for s in lipids_handled["lower"]:
			output_stat.write(str(s) + "	" + str(lipids_nff_sele["lower"][s].numberOfResidues()) + "	" + str(round(op_tailA_avg_frame["lower"][s][f_nb],2)) + "	" + str(round(op_tailA_avg_frame["lower"][s][f_nb],2)) + "	 " + str(round(op_both_avg_frame["lower"][s][f_nb],2)) + "\n")
		output_stat.write("\n")
		output_stat.write("std	nb	tail A	tail B	 both\n")
		output_stat.write("-------------------------------------\n")
		for s in lipids_handled["lower"]:
			output_stat.write(str(s) + "	" + str(lipids_nff_sele["lower"][s].numberOfResidues()) + "	" + str(round(op_tailA_std_frame["lower"][s][f_nb],2)) + "	" + str(round(op_tailA_std_frame["lower"][s][f_nb],2)) + "	 " + str(round(op_both_std_frame["lower"][s][f_nb],2)) + "\n")
		output_stat.close()

		#ff lipids
		#----------
		if args.selection_file_ff!="no":
			filename_details=os.getcwd() + '/' + str(args.output_folder) + '/2_snapshots/' + args.xtcfilename[:-4] + '_annotated_orderparam_' + str(int(t)).zfill(5) + 'ns_ff.stat'
			output_stat = open(filename_details, 'w')		
			output_stat.write("[lipid tail order parameters statistics - written by order_param v" + str(version_nb) + "]\n")
			output_stat.write("\n")
		
			#general info
			output_stat.write("1. membrane composition\n")
			output_stat.write(membrane_comp["upper"] + "\n")
			output_stat.write(membrane_comp["lower"] + "\n")
			tmp_string=str(lipids_handled["both"][0])
			for s in lipids_handled["both"][1:]:
				tmp_string+=", " + str(s)
			output_stat.write("\n")
			output_stat.write("2. lipid species processed: " + str(tmp_string) + "\n")
			output_stat.write("\n")
			output_stat.write("3. nb frames processed:	" + str(nb_frames_processed) + " (" + str(nb_frames_xtc) + " frames in xtc, step=" + str(args.frames_dt) + ")\n")
			output_stat.write("\n")
			output_stat.write("4. time: " + str(t) + "ns (frame " + str(f_nb) + "/" + str(nb_frames_xtc) + ")\n")
			output_stat.write("\n")
			output_stat.write("lipid orientation with bilayer normal\n")
			output_stat.write(" P2=1    : parallel\n")
			output_stat.write(" P2=0    : random\n")
			output_stat.write(" P2=-0.5 : orthogonal\n")
		
			#upper to lower
			if numpy.size(lipids_ff_u2l_index)>0:
				output_stat.write("\n")
				output_stat.write("upper to lower\n")
				output_stat.write("==============\n")
				output_stat.write("specie	resid	tail A	tail B  both\n")
				output_stat.write("-------------------------------------\n")
				for l in lipids_ff_u2l_index:
					output_stat.write(str(lipids_ff_info[l][0]) + "	" + str(lipids_ff_info[l][1]) + "	" + str(round(op_ff_tailA[l][f_nb],2)) + "	" + str(round(op_ff_tailB[l][f_nb],2)) + "	" + str(round(op_ff_both[l][f_nb],2)) + "\n")
			
			#lower to upper
			if numpy.size(lipids_ff_l2u_index)>0:
				output_stat.write("\n")
				output_stat.write("lower to upper\n")
				output_stat.write("==============\n")
				output_stat.write("specie	resid	tail A	tail B  both\n")
				output_stat.write("-------------------------------------\n")
				for l in lipids_ff_l2u_index:
					output_stat.write(str(lipids_ff_info[l][0]) + "	" + str(lipids_ff_info[l][1]) + "	" + str(round(op_ff_tailA[l][f_nb],2)) + "	" + str(round(op_ff_tailB[l][f_nb],2)) + "	" + str(round(op_ff_both[l][f_nb],2)) + "\n")
			output_stat.close()
	
	return
def write_frame_snapshot(f_nb, f_index,t):

	#store order parameter info in beta factor field: nff lipids
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			for r_index in range(0,lipids_nff_sele_nb[l][s]):
				lipids_selection_nff[l][s][r_index].set_bfactor(lipids_nff_op[l][s][r_index][f_index])
	#store order parameter info in beta factor field: ff lipids
	if args.selection_file_ff!="no":
		for l in range(0,lipids_ff_nb):
			lipids_selection_ff[l].set_bfactor(op_ff_both[l][f_nb])

	#case: gro file
	if args.xtcfilename=="no":
		all_atoms.write(os.getcwd() + '/' + str(args.output_folder) + '/2_snapshots/' + args.grofilename[:-4] + '_annotated_orderparam', format="PDB")

	#case: xtc file
	else:
		tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/2_snapshots/' + args.xtcfilename[:-4] + '_annotated_orderparam_' + str(int(t)).zfill(5) + 'ns.pdb'
		W=Writer(tmp_name, nb_atoms)
		W.write(all_atoms)
	
	return
def write_frame_annotation(f_nb, f_index,t):
	
	#create file
	if args.xtcfilename=="no":
		filename_details=os.getcwd() + "/" + str(args.output_folder) + '/2_snapshots/' + args.grofilename[:-4] + '_annotated_orderparam.txt'
	else:
		filename_details=os.getcwd() + "/" + str(args.output_folder) + '/2_snapshots/' + args.xtcfilename[:-4] + '_annotated_orderparam_' + str(int(t)).zfill(5) + 'ns.txt'
	output_stat = open(filename_details, 'w')		

	#output selection strings: nff lipids
	tmp_sele_string=""
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			for r_index in range(0,lipids_nff_sele_nb[l][s]):
				tmp_sele_string+="." + lipids_selection_nff_VMD_string[l][s][r_index]
	#output selection strings: ff lipids
	if args.selection_file_ff!="no":
		for l in range(0,lipids_ff_nb):
			tmp_sele_string+="." + lipids_selection_ff_VMD_string[l]
	output_stat.write(tmp_sele_string[1:] + "\n")
	
	#ouptut order param for each lipid for current frame: nff lipids
	tmp_ops="1"
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			for r_index in lipids_nff_op[l][s]:
				tmp_ops+=";" + str(round(lipids_nff_op[l][s][r_index][f_index],2))
	#ouptut order param for each lipid for current frame: ff lipids
	if args.selection_file_ff!="no":
		for l in range(0,lipids_ff_nb):
			tmp_ops+=";" + str(round(op_ff_both[l][f_nb],2))
	output_stat.write(tmp_ops + "\n")
	output_stat.close()

	return
def write_xtc_snapshots():
	#NB: - this will always output the first and final frame snapshots
	#    - it will also intermediate frames according to the -w option	

	loc_nb_frames_processed=0
	for ts in U.trajectory:

		#case: frames before specified time boundaries
		#---------------------------------------------
		if ts.time/float(1000)<args.t_start:
			progress='\r -skipping frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)

		#case: frames within specified time boundaries
		#---------------------------------------------
		elif ts.time/float(1000)>args.t_start and ts.time/float(1000)<args.t_end:
			progress='\r -writing snapshots...   frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)
			if ((ts.frame-1) % args.frames_dt)==0:
				if ((loc_nb_frames_processed) % args.frames_write_dt)==0 or loc_nb_frames_processed==nb_frames_processed-1:
					write_frame_stat(ts.frame, loc_nb_frames_processed, ts.time/float(1000))
					write_frame_snapshot(ts.frame, loc_nb_frames_processed, ts.time/float(1000))
					write_frame_annotation(ts.frame, loc_nb_frames_processed, ts.time/float(1000))
				loc_nb_frames_processed+=1
		
		#case: frames after specified time boundaries
		#--------------------------------------------
		elif ts.time/float(1000)>args.t_end:
			break

	print ''

	return
def write_xtc_annotation():
	
	#create file
	filename_details=os.getcwd() + '/' + str(args.output_folder) + '/3_VMD/' + args.xtcfilename[:-4] + '_annotated_orderparam_dt' + str(args.frames_dt) + '.txt'
	output_stat = open(filename_details, 'w')		

	#output selection strings
	#------------------------
	#nff lipids
	tmp_sele_string=""
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			for r_index in range(0,lipids_nff_sele_nb[l][s]):
				tmp_sele_string+="." + lipids_selection_nff_VMD_string[l][s][r_index]
	#ff lipids
	if args.selection_file_ff!="no":
		for l in range(0,lipids_ff_nb):
			tmp_sele_string+="." + lipids_selection_ff_VMD_string[l]
	output_stat.write(tmp_sele_string[1:] + "\n")
	
	#ouptut order param for each lipid
	#---------------------------------
	for frame in sorted(time_stamp.iterkeys()):
		tmp_ops=str(frame)
		frame_index=sorted(time_stamp.keys()).index(frame)
		#nff lipids
		for l in ["lower","upper"]:
			for s in lipids_handled[l]:
				for r_index in lipids_nff_op[l][s]:
					tmp_ops+=";" + str(round(lipids_nff_op[l][s][r_index][frame_index],2))
		#ff lipids
		if args.selection_file_ff!="no":
			for l in range(0,lipids_ff_nb):
				tmp_ops+=";" + str(round(op_ff_both[l][frame],2))
		output_stat.write(tmp_ops + "\n")
	output_stat.close()

	return

################################################################################################################################################
# ALGORITHM : Browse trajectory and process relevant frames
################################################################################################################################################

print "\nCalculating order parameters..."

#case: structure only
#====================
if args.xtcfilename=="no":
	time_stamp[1]=0
	calculate_order_parameters(1)

#case: browse xtc frames
#=======================
else:
	for ts in U.trajectory:
		
		#case: frames before specified time boundaries
		#---------------------------------------------
		if ts.time/float(1000)<args.t_start:
			progress='\r -skipping frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)

		#case: frames within specified time boundaries
		#---------------------------------------------
		elif ts.time/float(1000)>args.t_start and ts.time/float(1000)<args.t_end:
			progress='\r -processing frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '                      '  
			sys.stdout.flush()
			sys.stdout.write(progress)
			if ((ts.frame-1) % args.frames_dt)==0:						
				nb_frames_processed+=1
				time_stamp[ts.frame]=ts.time/float(1000)
				get_z_coords(ts.frame)
				calculate_order_parameters(ts.frame)
			
		#case: frames after specified time boundaries
		#---------------------------------------------
		elif ts.time/float(1000)>args.t_end:
			break
	print ''

################################################################################################################################################
# CALCULATE STATISTICS
################################################################################################################################################

print "\nCalculating statistics..."
calculate_stats()
if args.nb_smoothing>1:
	smooth_data()
		
################################################################################################################################################
# PRODUCE OUTPUTS
################################################################################################################################################

print "\nWriting outputs..."

#case: gro file
#==============
if args.xtcfilename=="no":
	print " -writing statistics..."
	write_frame_stat(1,"all","all")
	print " -writing annotated pdb..."
	write_frame_snapshot(1,0,0)
	write_frame_annotation(1,0,0)

#case: xtc file
#==============
else:
	#writing statistics
	print " -writing statistics..."
	write_frame_stat(1,"all","all")
	#output cluster snapshots
	write_xtc_snapshots()
	#write annotation files for VMD
	print " -writing VMD annotation files..."
	write_xtc_annotation()
	#write xvg and graphs
	print " -writing xvg and graphs..."
	write_op_nff_xvg()
	graph_op_nff_xvg()	
	if args.nb_smoothing>1:
		write_op_nff_xvg_smoothed()
		graph_op_nff_xvg_smoothed()		
	if args.selection_file_ff!="no":
		write_op_ff_xvg()
		graph_op_ff_xvg()	
		if args.nb_smoothing>1:
			write_op_ff_xvg_smoothed()
			graph_op_ff_xvg_smoothed()	
			
#exit
#====
print "\nFinished successfully! Check output in ./" + args.output_folder + "/"
print ""
sys.exit(0)

export PETSC_DIR=/home/rlchen/soft/petsc-3.5.2
export PETSC_ARCH=64bit-debug

CFLAGS    = $(CPPFLAGS)  -D__MYSDIR__='"$(LOCDIR)"'

ALL: AeroSim

AeroSim.o SetupMeshes.o SetupFunctions.o SetupJacobian.o Functions.o : AeroSim.h

AeroSim: AeroSim.o SetupMeshes.o SetupFunctions.o SetupJacobian.o Functions.o \
	src/plexexodusii.o src/plexgeometry.o src/plex.o \
	src/plexfem.o src/plexsubmesh.o src/plexlabel.o \
	src/dtfv.o src/dm.o src/vsectionis.o src/plexvtk.o chkopts
	-${CLINKER} -o AeroSim AeroSim.o SetupMeshes.o SetupFunctions.o SetupJacobian.o Functions.o \
	src/plexexodusii.o src/plexgeometry.o src/plex.o \
	src/plexfem.o src/plexsubmesh.o src/plexlabel.o \
	src/dtfv.o src/dm.o src/vsectionis.o src/plexvtk.o ${PETSC_TS_LIB}
	${RM} *.o

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test


runtest:
	-@${MPIEXEC} -n 4 ./AeroSim -show_partition -f ./../meshes/3DHillFVMHex1.exo -overlap 1 \
	-reconstruct -time_integral_method 0 -final_time 5.0 -dt 0.001 \
	-includeenergy -max_time_its 2000 -ts_view -myownexplicitmethod -PressureFlux -Explicit_RK2 \
	-output_solution -solutionfile solution -steps_output 50 -inflow_u 10.0 -viscosity 0.0 -k 0.026 \
	-R 8.3 -T0 0.8 -T1 0.85 -adiabatic 1.4 \
	-limiter minmod -ts_max_steps 2000 -Euler \

runbench:
	-@${MPIEXEC} -n 4 ./AeroSim -show_partition -f ./../meshes/3DCouetteFVM4096.exo -overlap 1 \
	-reconstruct -time_integral_method 0 -final_time 5.0 -dt 0.001 \
	-includeenergy -max_time_its 1000 -ts_view -myownexplicitmethod \
	-output_solution -solutionfile solution -steps_output 50 -inflow_u 1.0 -viscosity 0.0 -k 0.026 \
	-R 8.3 -T0 0.8 -T1 0.85 -adiabatic 1.4 -benchmark_couette \
	-limiter minmod -ts_max_steps 1000 -Euler \

rundriven:
	-@${MPIEXEC} -n 4 ./AeroSim -show_partition -f ./../meshes/3DDrivencavityFVM512.exo -overlap 1 \
	-reconstruct -time_integral_method 0 -final_time 5.0 -dt 0.001 \
	-includeenergy -max_time_its 2000 -myownexplicitmethod -PressureFlux \
	-output_solution -solutionfile solution -steps_output 50 -inflow_u 1.0 -viscosity 0.0 -k 0.026 \
	-R 8.3 -T0 0.8 -T1 0.85 -adiabatic 1.4 \
	-limiter minmod -ts_max_steps 2000 -Euler \

runimplicit:
	-@${MPIEXEC} -n 12 ./AeroSim -show_partition -f meshes/3DCouetteFVM512.exo -overlap 1 \
	-ksp_type gmres -ksp_pc_side right -pc_type asm -pc_asm_overlap 1 -sub_pc_type ilu -sub_pc_factor_levels 2 \
	-ksp_gmres_restart 100 -ksp_rtol 1.e-4 -ksp_max_it 100 \
	-reconstruct -time_integral_method 1 -initial_time 0.0 -final_time 0.01 -dt 0.001 -timestep 0 \
	-includeenergy -snes_rtol 1.e-6 -snes_max_it 0 \
	-R 8.3 -T0 0.8 -T1 0.85 -adiabatic 1.4 -PressureFlux -benchmark_couette \
	-output_solution -inflow_u 1.0 -viscosity 0.0 -k 0.026 -limiter none -fd_jacobian -Euler



#        -ts_monitor_lg_solution -ts_monitor_lg_error -ts_view

# cubit_hex16.exo cubit_tet60.exo   -malloc_debug -malloc_dump


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
	-@${MPIEXEC} -n 1 ./AeroSim -show_partition -f ./../meshes/3DCouetteFVM8.exo -overlap 1 \
	-reconstruct -time_integral_method 0 -final_time 500.0 -dt 0.001 \
	-includeenergy -max_time_its 0 -ts_view -myownexplicitmethod -PressureFlux \
	-output_solution -solutionfile solution -steps_output 50 -inflow_u 1.0 -inflow_v 1.0 -inflow_w 1.0 -viscosity 0.0 -k 0.026 \
	-R 8.3 -T0 0.8 -T1 0.85 -adiabatic 1.4 \
	-limiter minmod -ts_max_steps 2000 -Euler -RiemannSolver rlchen -CFL 0.9 \

runtest1:
	-@${MPIEXEC} -n 4 ./AeroSim -show_partition -f ./../meshes/3DCubeFVMHex1.exo -overlap 1 \
	-time_integral_method 0 -final_time 500.0 -dt 0.001 \
	-includeenergy -max_time_its 5000 -ts_view -myownexplicitmethod -PressureFlux \
	-output_solution -solutionfile results/solution -steps_output 1 -inflow_u 340.0 -viscosity 0.01 -k 0.026 \
	-R 8.3 -T0 0.8 -T1 0.85 -adiabatic 1.4 \
	-limiter minmod -ts_max_steps 2000 -RiemannSolver Rusanov -CFL 0.9 -Euler \

runbench:
	-@${MPIEXEC} -n 4 ./AeroSim -show_partition -f ./../meshes/3DCouetteFVM8.exo -overlap 1 \
	-reconstruct -time_integral_method 0 -final_time 1000.0 -dt 0.001 \
	-includeenergy -max_time_its 100 -ts_view -ts_monitor -myownexplicitmethod -PressureFlux \
	-output_solution -solutionfile solution -steps_output 50 -inflow_u 1.0 -viscosity 0.0 -k 0.026 \
	-R 8.3 -T0 0.8 -T1 0.85 -adiabatic 1.4 -benchmark_couette \
	-limiter minmod -ts_max_steps 2000 -Euler -RiemannSolver rlchen -CFL 0.9 \

rundriven:
	-@${MPIEXEC} -n 4 ./AeroSim -show_partition -f ./../meshes/3DDrivencavityFVM512.exo -overlap 1 \
	-reconstruct -time_integral_method 0 -final_time 5.0 -dt 0.001 \
	-includeenergy -max_time_its 2000 -myownexplicitmethod -PressureFlux \
	-output_solution -solutionfile solution -steps_output 50 -inflow_u 1.0 -viscosity 0.0 -k 0.026 \
	-R 8.3 -T0 0.8 -T1 0.85 -adiabatic 1.4 \
	-limiter minmod -ts_max_steps 2000 -Euler -RiemannSolver David \

SNESOPTIONS = -snes_rtol 1.e-6 -snes_max_it 0
KSPOPTIONS  = -ksp_type gmres -ksp_pc_side right -ksp_gmres_restart 100 -ksp_rtol 1.e-4 -ksp_max_it 100 -ksp_monitor
PCOPTIONS   = -pc_type asm -pc_asm_overlap 1 -sub_pc_type lu -sub_pc_factor_levels 2
runimplicit:
	-@${MPIEXEC} -n 4 ./AeroSim -show_partition -f ./../meshes/3DCouetteFVM8.exo \
	${SNESOPTIONS} ${KSPOPTIONS} ${PCOPTIONS} \
	-time_integral_method 1 -initial_time 0.0 -final_time 0.01 -dt 0.001 -timestep 0 \
	-includeenergy -reconstruct -overlap 1 \
	-R 8.3 -T0 0.8 -T1 0.85 -adiabatic 1.4 -PressureFlux -benchmark_couette \
	-inflow_u 1.0 -viscosity 0.0 -k 0.026 -limiter none -fd_jacobian -Euler -RiemannSolver rlchen \
	-output_solution -solutionfile results/solution



#        -ts_monitor_lg_solution -ts_monitor_lg_error -ts_view

# cubit_hex16.exo cubit_tet60.exo   -malloc_debug -malloc_dump


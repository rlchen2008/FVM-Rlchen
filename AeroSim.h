
#define DIM 3                   /* Geometric dimension */
#define EXPLICITMETHOD 0        /*explicit method for the time integral*/
#define IMPLICITMETHOD 1        /*implicit method for the time integral*/
#define MYTOLERANCE 1.e-20        /* The tolerance for a nonzero number, that is if abs(f)<MYTOLERANCE, then let f = 0.0  */

/* time step specification */
#define TIMESTEP_STEADY_STATE 0
#define TIMESTEP_BEULER 1
#define TIMESTEP_BDF 2
#define TIMESTEP_TRAPEZOIDAL 3

#define ALEN(a) (sizeof(a)/sizeof((a)[0]))
#define Mydebug printf("\n I am at the line %d in file %s\n", __LINE__, __FILE__);

PetscFunctionList LimitList;

/* Represents continuum physical equations. */
typedef struct _n_Physics *Physics;

/* Physical model includes boundary conditions, initial conditions, and functionals of interest. It is
 * discretization-independent, but its members depend on the scenario being solved. */
typedef struct _n_Model *Model;

/* 'User' implements a discretization of a continuous model. */
typedef struct _n_User *User;

/*Algebra implements the vectors, matrice used in the solution*/
typedef struct _n_Algebra *Algebra;

typedef PetscErrorCode (*RiemannFunction)(Physics,const PetscReal*,const PetscReal*,const PetscReal*,const PetscReal*,PetscReal*);
typedef PetscErrorCode (*SolutionFunction)(Model,PetscReal,const PetscReal*,PetscReal*,void*);
typedef PetscErrorCode (*FunctionalFunction)(Model,PetscReal,const PetscReal*,const PetscReal*,PetscReal*,void*);
typedef PetscErrorCode (*SetupFields)(Physics,PetscSection);
typedef PetscErrorCode (*BoundaryFunction)(Model,PetscReal,const PetscReal*,const PetscReal*,const PetscReal*,PetscReal*,void*);


struct FieldDescription {
  const char *name;
  PetscInt dof;
};

struct _n_Physics {
  PetscInt        dof;          /* number of degrees of freedom per cell */
  PetscInt        nfields;
  const struct FieldDescription *field_desc;
};

struct _n_Model {
  MPI_Comm         comm;        /* Does not do collective communicaton, but some error conditions can be collective */
  Physics          physics;
};

struct _n_User {
  PetscReal      (*Limit)(PetscReal); /*The limiter function*/
  PetscReal      (*LimitGrad)(PetscReal); /*The derivative of the limiter function*/
  PetscBool      reconstruct;
  PetscInt       numGhostCells, numSplitFaces;
  PetscInt       cEndInterior; /* First boundary ghost cell */
  Vec            cellgeom, facegeom;
  DM             dmGrad;
  PetscReal      minradius;
  PetscInt       vtkInterval;   /* For monitor */
  Model          model;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscBool      show_partition; /*Show the partition of the mesh*/
  PetscInt       overlap; /*Number of cells to overlap partitions*/
  DM             dm;
  SNES           snes; /* nonlinear solver object*/
  Algebra        algebra;
  PetscBool      includeenergy; /*inculde the energy field*/
  PetscInt       TimeIntegralMethod; /*the time integral method, 0 for the explicit and 1 for the implicit method*/
  PetscBool      output_solution; /*Output the solution*/
  PetscBool      JdiffP; /*Use different matrix for the Jacobian and preconditioner*/
  PetscInt       timestep; /*scheme for the implicit time integral: backward Euler, BDF2, ...*/
  PetscReal      dt; /*time step size*/
  PetscBool      fd_jacobian;
  PetscBool      fd_jacobian_color;

  PetscReal      current_time;
  PetscReal      initial_time;
  PetscReal      final_time;
  PetscInt       current_step;
  PetscBool      second_order; /*Use the second order scheme*/

  PetscReal      KDof; /*kinematic dof, for the monoatomic gas, KDof = 3, for a diatomic gas, KDof = 5*/
  PetscReal      adiabatic; /*the heat capacity ratio or called adiabatic*/
  /*
     The relation of the kinematic dof and the heat capacity ratio for the ideal gas is
     adiabatic = 1 + 2/KDof
  */
  PetscReal      R; /*The gas constant which is 8.3144621 J K^{-1} mol^{-1} for the state equation*/
  PetscReal      viscosity;
  PetscReal      k; /*the thermal conductivity coefficient which is 0.026 for the air at 300K*/
  PetscReal      inflow_u; /*The inflow velocity component u*/
  PetscReal      inflow_v; /*The inflow velocity component v*/
  PetscReal      inflow_w; /*The inflow velocity component w*/
  PetscBool      myownexplicitmethod;
  PetscBool      PressureFlux; /*Use the flux style for the pressure terms*/
  PetscBool      benchmark_couette; /*For the Couette benchmark test*/
  PetscInt       max_time_its;
  PetscBool      Explicit_RK2; /*Use second order Runge-Kutta method*/
  PetscBool      Explicit_RK4; /*Use 4th order Runge-Kutta method*/
  PetscReal      T0, T1;
  PetscBool      Euler; /*Use Euler equation*/
  char           solutionfile[PETSC_MAX_PATH_LEN];/*the file name for the solution output*/
  PetscInt       steps_output; /* the number of time steps between two outputs */
  char           RiemannSolver[PETSC_MAX_PATH_LEN];
  PetscReal      CFL; // The Courant number coefficient
  PetscBool      orthogonal_correct; // The orthogonal correction
  PetscBool      simple_diffusion; // Using a simple diffusion version
  struct {
    PetscReal *flux;
    PetscReal *state0;
    PetscReal *state1;
  } work;
};

struct _n_Algebra {
  Vec oldsolution; /* initial guess or previous solution */
  Vec oldoldsolution; /* initial guess or previous solution */
  Vec exactsolution; /* initial guess or previous solution */
  Vec f; /* for the nonlinear function */

  Vec solution;

  Vec fn;
  Vec oldfn;

  Vec f_local; /* local values of nonlinear function */
  Vec oldsolution_local; /* local solution vector at previous time step */


  Mat A,J,M,P; /* M is a mass matrix, A is everything else, J holds the Jacobian, P is the preconditioner matrix */
};

typedef struct {
  PetscReal vals[0];
  /*Zero-length array,just like a pointer and vals[0] points to r, vals[1] points to u[0], ...*/
  PetscReal r; /*the density \rho*/
  PetscReal ru[DIM]; /*the density momontum u*/
  PetscReal rE; /*the density total energy, that is the total energy in per unit volume*/
} Node;

static const struct FieldDescription PhysicsFields_Euler[] = {{"Density",1},{"Momentum",DIM},{"Energy",1},{NULL,0}};
static const struct FieldDescription PhysicsFields_Partial[] = {{"Density",1},{"Momentum",DIM},{NULL,0}};
static const struct FieldDescription PhysicsFields_Full[] = {{"Density",1},{"Momentum",DIM},{"Energy",1},{NULL,0}};

extern PetscErrorCode ModelSolutionSetDefault(Model,SolutionFunction,void*);
extern PetscErrorCode ModelFunctionalRegister(Model,const char*,PetscInt*,FunctionalFunction,void*);
extern PetscErrorCode OutputVTK(DM,const char*,PetscViewer*);
extern PetscErrorCode CreatePartitionVec(DM dm, DM *dmCell, Vec *partition);
extern PetscErrorCode ConstructGeometry(DM dm, Vec *facegeom, Vec *cellgeom, User user);
extern PetscErrorCode BuildLeastSquares(DM dm,PetscInt cEndInterior,DM dmFace,PetscReal *fgeom,DM dmCell,PetscReal *cgeom);
extern PetscErrorCode PseudoInverseGetWorkRequired(PetscInt maxFaces,PetscInt *work);
extern PetscErrorCode IsExteriorGhostFace(DM dm,PetscInt face,PetscBool *isghost);
extern PetscErrorCode PseudoInverseSVD(PetscInt m,PetscInt mstride,PetscInt n,PetscReal *A,PetscReal *Ainv,PetscReal *tau,PetscInt worksize,PetscReal *work);
extern PetscErrorCode PetscFVLeastSquaresPseudoInverse_Static(PetscInt m,PetscInt mstride,PetscInt n,PetscScalar *A,PetscScalar *Ainv,PetscScalar *tau,PetscInt worksize,PetscScalar *work);
extern PetscErrorCode PetscFVLeastSquaresPseudoInverseSVD_Static(PetscInt m,PetscInt mstride,PetscInt n,PetscScalar *A,PetscScalar *Ainv,PetscScalar *tau,PetscInt worksize,PetscScalar *work);
extern PetscErrorCode PseudoInverse(PetscInt m,PetscInt mstride,PetscInt n,PetscReal *A,PetscReal *Ainv,PetscReal *tau,PetscInt worksize,PetscReal *work);
extern PetscErrorCode ConstructGeometryFVM(Vec *facegeom, Vec *cellgeom, User user);
extern PetscErrorCode CreateMesh(MPI_Comm comm, User user);
extern PetscErrorCode LoadOptions(MPI_Comm comm, User user);
extern PetscErrorCode SetUpLocalSpace(User user);
extern PetscErrorCode SetInitialCondition(DM dm, Vec X, User user);
extern PetscErrorCode SetInitialGuess(DM dm, Vec X, User user);
extern PetscErrorCode Pressure_PG(User user,const Node *x,PetscReal *p);
extern PetscErrorCode SpeedOfSound_PG(User user,const Node *x,PetscReal *c);
extern PetscErrorCode ConvectionFlux(User user,const PetscReal *n,const Node *x,Node *f);
extern PetscErrorCode DiffusionFlux(User user, const PetscReal *cgradL, const PetscReal *cgradR, const PetscReal *fgc,
                                     const PetscReal *cgcL, const PetscReal *cgcR,
                                     const PetscReal *n, const Node *xL, const Node *xR, Node *f);
extern PetscErrorCode RiemannSolver(User user, const PetscReal *cgradL, const PetscReal *cgradR,
                                             const PetscReal *fgc, const PetscReal *cgcL, const
                                             PetscReal *cgcR, const PetscReal *n, const PetscReal *xL, const PetscReal *xR,
                                             PetscReal *fluxcon, PetscReal *fluxdiff);
extern PetscErrorCode RiemannSolver_Rusanov_Jacobian(User user, const PetscReal *cgradL, const PetscReal *cgradR,
                                             const PetscReal *fgc, const PetscReal *cgcL, const
                                             PetscReal *cgcR, const PetscReal *n, const PetscReal *xL, const PetscReal *xR,
                                             PetscReal *fluxcon, PetscReal *fluxdiff);
extern PetscErrorCode ModelFunctionalRegister(Model mod,const char *name,PetscInt *offset,FunctionalFunction func,void *ctx);
extern PetscErrorCode MonitorVTK(TS ts,PetscInt stepnum,PetscReal time,Vec X,void *ctx);
extern PetscErrorCode ModelBoundaryFind(Model mod,PetscInt id,BoundaryFunction *bcFunc,void **ctx);
extern PetscErrorCode ApplyBC(DM dm, PetscReal time, Vec locX, User user);
extern PetscErrorCode FormFunction(SNES snes, Vec y, Vec out, void *ctx);
extern PetscErrorCode FormTimeStepFunction(User user, Algebra algebra, Vec in, Vec out);
extern PetscErrorCode MyRHSFunction(TS ts,PetscReal time,Vec in,Vec out,void *ctx);
extern PetscErrorCode FormMassTimeStepFunction(User user, Algebra algebra, Vec in, Vec out, PetscBool rebuild);
extern PetscErrorCode FormJacobian(SNES snes, Vec g, Mat jac, Mat B, void *ctx);
extern PetscErrorCode InitialCondition(PetscReal time, const PetscReal *x, PetscReal *u, User user);
extern PetscErrorCode InitialGuess(PetscReal time, const PetscReal *x, PetscReal *u, User user);
extern PetscErrorCode SolveSteadyState(void* ctx);
extern PetscErrorCode SolveTimeDependent(void* ctx);
extern PetscErrorCode CaculateLocalMassFunction(DM dm,Vec locX,Vec F,User user);
extern PetscErrorCode CaculateLocalFunction_LS(DM dm,DM dmFace,DM dmCell,PetscReal time,Vec locX,Vec F,User user);
extern PetscErrorCode CaculateLocalFunction_Upwind(DM dm,DM dmFace,DM dmCell,PetscReal time,Vec locX,Vec F,User user);
extern PetscErrorCode LimiterSetup(User user);
extern PetscErrorCode Pressure_Full(User user,const Node *x,PetscReal *p);
extern PetscErrorCode Pressure_Partial(User user,const Node *x,PetscReal *p);
extern PetscErrorCode Energy(User user,const Node *x,PetscReal *e);
extern PetscErrorCode ConstructCellCentriodGradient(DM dm,DM dmFace,DM dmCell,PetscReal time,Vec locX,Vec F,User user);
extern PetscErrorCode MonitorFunction(SNES snes, PetscInt its, double norm, void *dctx);
extern PetscErrorCode BoundaryInflow(PetscReal time, const PetscReal *c, const PetscReal *n,
                                      const PetscReal *xI, PetscReal *xG, User user);
extern PetscErrorCode BoundaryOutflow(PetscReal time, const PetscReal *c, const PetscReal *n,
                                      const PetscReal *xI, PetscReal *xG, User user);
extern PetscErrorCode BoundaryWallflow(PetscReal time, const PetscReal *c, const PetscReal *n,
                                       const PetscReal *xI, PetscReal *xG, User user);
extern PetscErrorCode SetupJacobian(DM dm, Vec X, Mat jac, Mat B, void *ctx);
extern PetscErrorCode ComputeJacobian_LS(DM dm, Vec locX, PetscInt cell, PetscReal CellValues[], void *ctx);
extern PetscErrorCode ComputeJacobian_Upwind(DM dm, Vec locX, PetscInt cell, PetscReal CellValues[], void *ctx);
extern PetscErrorCode GradientGradientJacobian(DM dm, Vec locX, PetscReal elemMat[], void *ctx);
extern PetscErrorCode DMPlexGetIndex(DM dm, PetscSection section, PetscSection globalSection,
                                      PetscInt point, PetscInt *NumOfIndices, PetscInt indices[]);
extern PetscErrorCode ConstructCellCentriodGradientJacobian(DM dm,DM dmFace,DM dmCell,PetscReal time,Vec locX, PetscInt cell, PetscReal CellValues[],User user);
extern PetscErrorCode CaculateLocalSourceTerm(DM dm, Vec locX, Vec F, User user);

extern PetscReal DotDIM(const PetscReal *x,const PetscReal *y);
extern PetscReal NormDIM(const PetscReal *x);
extern void axDIM(const PetscReal a,PetscReal *x);
extern void waxDIM(const PetscReal a,const PetscReal *x, PetscReal *w);
extern void NormalSplitDIM(const PetscReal *n,const PetscReal *x,PetscReal *xn,PetscReal *xt);
extern PetscReal Dot2(const PetscReal *x,const PetscReal *y);
extern PetscReal Norm2(const PetscReal *x);
extern void Normalize2(PetscReal *x);
extern void Waxpy2(PetscReal a,const PetscReal *x,const PetscReal *y,PetscReal *w);
extern void Scale2(PetscReal a,const PetscReal *x,PetscReal *y);
extern void WaxpyD(PetscInt dim, PetscReal a, const PetscReal *x, const PetscReal *y, PetscReal *w);
extern PetscReal DotD(PetscInt dim, const PetscReal *x, const PetscReal *y);
extern PetscReal NormD(PetscInt dim, const PetscReal *x);
extern void NormalSplit(const PetscReal *n,const PetscReal *x,PetscReal *xn,PetscReal *xt);
extern PetscReal SourceRho(User user, const PetscReal *cgrad, const PetscReal *x, const PetscReal *xcoord);
extern PetscReal SourceU(User user, const PetscReal *cgrad, const PetscReal *x, const PetscReal *xcoord);
extern PetscReal SourceV(User user, const PetscReal *cgrad, const PetscReal *x, const PetscReal *xcoord);
extern PetscReal SourceW(User user, const PetscReal *cgrad, const PetscReal *x, const PetscReal *xcoord);
extern PetscReal SourceE(User user, const PetscReal *cgrad, const PetscReal *x, const PetscReal *xcoord);

extern PetscReal Limit_Zero(PetscReal f);
extern PetscReal Limit_None(PetscReal f);
extern PetscReal Limit_Minmod(PetscReal f);
extern PetscReal Limit_VanLeer(PetscReal f);
extern PetscReal Limit_VanAlbada(PetscReal f);
extern PetscReal Limit_Sin(PetscReal f);
extern PetscReal Limit_Superbee(PetscReal f);
extern PetscReal Limit_MC(PetscReal f);

extern PetscReal Limit_ZeroGrad(PetscReal f);
extern PetscReal Limit_NoneGrad(PetscReal f);
extern PetscReal Limit_MinmodGrad(PetscReal f);
extern PetscReal Limit_VanLeerGrad(PetscReal f);
extern PetscReal Limit_VanAlbadaGrad(PetscReal f);
extern PetscReal Limit_SinGrad(PetscReal f);
extern PetscReal Limit_SuperbeeGrad(PetscReal f);
extern PetscReal Limit_MCGrad(PetscReal f);

extern PetscErrorCode TSMonitorFunctionError(TS ts,PetscInt step,PetscReal ptime,Vec u,void *ctx);
extern PetscErrorCode ComputeExactSolution(DM dm, PetscReal time, Vec locX, User user);
extern PetscErrorCode ExactSolution(PetscReal time, const PetscReal *c, PetscReal *xc, User user);
extern PetscErrorCode ReformatSolution(Vec solution, Vec solution_unscaled, User user);
extern PetscErrorCode SNESComputeJacobianDefaultDebug(SNES snes,Vec x1,Mat J,Mat B,void *ctx);

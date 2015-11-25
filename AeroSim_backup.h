
#define DIM 2                   /* Geometric dimension */
#define ALEN(a) (sizeof(a)/sizeof((a)[0]))

/* Represents continuum physical equations. */
typedef struct _n_Physics *Physics;

/* Physical model includes boundary conditions, initial conditions, and functionals of interest. It is
 * discretization-independent, but its members depend on the scenario being solved. */
typedef struct _n_Model *Model;

/* 'User' implements a discretization of a continuous model. */
typedef struct _n_User *User;

typedef void (*RiemannFunction)(const PetscReal*,const PetscReal*,const PetscScalar*,const PetscScalar*,PetscScalar*,void*);
typedef PetscErrorCode (*SolutionFunction)(Model,PetscReal,const PetscReal*,PetscScalar*,void*);
typedef PetscErrorCode (*FunctionalFunction)(Model,PetscReal,const PetscReal*,const PetscScalar*,PetscReal*,void*);
typedef PetscErrorCode (*SetupFields)(Physics,PetscSection);


struct FieldDescription {
  const char *name;
  PetscInt dof;
};

typedef struct _n_FunctionalLink *FunctionalLink;
struct _n_FunctionalLink {
  char               *name;
  FunctionalFunction func;
  void               *ctx;
  PetscInt           offset;
  FunctionalLink     next;
};

struct _n_Physics {
  RiemannFunction riemann;
  PetscInt        dof;          /* number of degrees of freedom per cell */
  PetscReal       maxspeed;     /* kludge to pick initial time step, need to add monitoring and step control */
  void            *data;
  PetscInt        nfields;
  const struct FieldDescription *field_desc;
};

struct _n_Model {
  MPI_Comm         comm;        /* Does not do collective communicaton, but some error conditions can be collective */
  Physics          physics;
  FunctionalLink   functionalRegistry;
  PetscInt         maxComputed;
  PetscInt         numMonitored;
  FunctionalLink   *functionalMonitored;
  PetscInt         numCall;
  FunctionalLink   *functionalCall;
  SolutionFunction solution;
  void             *solutionctx;
  PetscReal        maxspeed;    /* estimate of global maximum speed (for CFL calculation) */
};

struct _n_User {
  PetscInt numSplitFaces;
  PetscInt vtkInterval;   /* For monitor */
  Model    model;
};

extern PetscErrorCode ModelSolutionSetDefault(Model,SolutionFunction,void*);
extern PetscErrorCode ModelFunctionalRegister(Model,const char*,PetscInt*,FunctionalFunction,void*);
extern PetscErrorCode OutputVTK(DM,const char*,PetscViewer*);

extern PetscScalar DotDIM(const PetscScalar *x,const PetscScalar *y);
extern PetscReal NormDIM(const PetscScalar *x);
extern void axDIM(const PetscScalar a,PetscScalar *x);
extern void waxDIM(const PetscScalar a,const PetscScalar *x, PetscScalar *w);
extern void NormalSplitDIM(const PetscReal *n,const PetscScalar *x,PetscScalar *xn,PetscScalar *xt);
extern PetscScalar Dot2(const PetscScalar *x,const PetscScalar *y);
extern PetscReal Norm2(const PetscScalar *x);
extern void Normalize2(PetscScalar *x);
extern void Waxpy2(PetscScalar a,const PetscScalar *x,const PetscScalar *y,PetscScalar *w);
extern void Scale2(PetscScalar a,const PetscScalar *x,PetscScalar *y);
extern void WaxpyD(PetscInt dim, PetscScalar a, const PetscScalar *x, const PetscScalar *y, PetscScalar *w);
extern PetscScalar DotD(PetscInt dim, const PetscScalar *x, const PetscScalar *y);
extern PetscReal NormD(PetscInt dim, const PetscScalar *x);
extern void NormalSplit(const PetscReal *n,const PetscScalar *x,PetscScalar *xn,PetscScalar *xt);








#include <petscts.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <petscblaslapack.h>

#include "AeroSim.h"

//static PetscFunctionList LimitList;

#undef __FUNCT__
#define __FUNCT__ "LimiterSetup"
/*@
 * Limiters given in symmetric form following Berger, Aftosmis, and Murman 2005
 *
 * The classical flux-limited formulation is psi(r) where
 *
 * r = (u[0] - u[-1]) / (u[1] - u[0])
 *
 * The second order TVD region is bounded by
 *
 * psi_minmod(r) = min(r,1)      and        psi_superbee(r) = min(2, 2r, max(1,r))
 *
 * where all limiters are implicitly clipped to be non-negative. A more convenient slope-limited form is psi(r) =
 * phi(r)(r+1)/2 in which the reconstructed interface values are
 *
 * u(v) = u[0] + phi(r) (grad u)[0] v
 *
 * where v is the vector from centroid to quadrature point. In these variables, the usual limiters become
 *
 * phi_minmod(r) = 2 min(1/(1+r),r/(1+r))   phi_superbee(r) = 2 min(2/(1+r), 2r/(1+r), max(1,r)/(1+r))
 *
 * For a nicer symmetric formulation, rewrite in terms of
 *
 * f = (u[0] - u[-1]) / (u[1] - u[-1])
 *
 * where r(f) = f/(1-f). Not that r(1-f) = (1-f)/f = 1/r(f) so the symmetry condition
 *
 * phi(r) = phi(1/r)
 *
 * becomes
 *
 * w(f) = w(1-f).
 *
 * The limiters below implement this final form w(f). The reference methods are
 *
 * w_minmod(f) = 2 min(f,(1-f))             w_superbee(r) = 4 min((1-f), f)
 *
@*/
PetscErrorCode LimiterSetup(User user)
{
  PetscErrorCode ierr;
  char           limitname[256] = "minmod";
  char           limitnameGrad[256] = "minmodGrad";

  PetscFunctionBeginUser;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"The limiter Options","");CHKERRQ(ierr);

  ierr = PetscFunctionListAdd(&LimitList,"zero"              ,Limit_Zero);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"none"              ,Limit_None);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"minmod"            ,Limit_Minmod);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"vanleer"           ,Limit_VanLeer);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"vanalbada"         ,Limit_VanAlbada);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"sin"               ,Limit_Sin);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"superbee"          ,Limit_Superbee);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"mc"                ,Limit_MC);CHKERRQ(ierr);

  ierr = PetscFunctionListAdd(&LimitList,"zeroGrad"              ,Limit_ZeroGrad);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"noneGrad"              ,Limit_NoneGrad);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"minmodGrad"            ,Limit_MinmodGrad);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"vanleerGrad"           ,Limit_VanLeerGrad);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"vanalbadaGrad"         ,Limit_VanAlbadaGrad);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"sinGrad"               ,Limit_SinGrad);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"superbeeGrad"          ,Limit_SuperbeeGrad);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&LimitList,"mcGrad"                ,Limit_MCGrad);CHKERRQ(ierr);

  ierr = PetscOptionsFList("-limiter","Limiter to apply to reconstructed solution","",
                            LimitList,limitname,limitname,sizeof(limitname), NULL);CHKERRQ(ierr);

  ierr = PetscFunctionListFind(LimitList,limitname,&user->Limit);CHKERRQ(ierr);
  sprintf(limitnameGrad,"%sGrad", limitname);
  ierr = PetscFunctionListFind(LimitList,limitnameGrad,&user->LimitGrad);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Limiter type (Grad): %s (%s)\n", limitname, limitnameGrad);

  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/************************Limiter Zero****************************************/
PetscReal Limit_Zero(PetscReal f)
{
  return 0;
}
PetscReal Limit_ZeroGrad(PetscReal f)
{
  return 0;
}

/************************Limiter None******************************************/
PetscReal Limit_None(PetscReal f)
{
  return 1;
}
PetscReal Limit_NoneGrad(PetscReal f)
{
  return 0;
}

/************************Limiter Minmod****************************************/
PetscReal Limit_Minmod(PetscReal f)
{
  return PetscMax(0,PetscMin(f,(1-f))*2);
}
PetscReal Limit_MinmodGrad(PetscReal f)
{ /*Compute the direvative of function PetscMax(0,PetscMin(f,(1-f))*2)*/
  PetscScalar result = 0.0;
  if (f < 1-f){
    if (f < 0.0) {
      result = 0.0;
    }else{
      result = 2.0;
    }
  }else{
    if(1-f < 0.0){
      result = 0.0;
    }else{
      result = -2.0;
    }
  }
  return result;
}

/************************Limiter VanLeer**************************************/
PetscReal Limit_VanLeer(PetscReal f)
{
  return PetscMax(0,4*f*(1-f));
}
PetscReal Limit_VanLeerGrad(PetscReal f)
{ /*Compute the direvative of function PetscMax(0,4*f*(1-f))*/
  PetscScalar result = 0.0;
  if(f*(1-f)<0.0){
    result = 0.0;
  }else{
    result = - (4.0 - 8.0*f);
  }
  return result;
}

/************************Limiter VanAlbada************************************/
PetscReal Limit_VanAlbada(PetscReal f)
{
  return PetscMax(0, 2*f*(1-f) / (PetscSqr(f) + PetscSqr(1-f)));
}
PetscReal Limit_VanAlbadaGrad(PetscReal f)
{ /*Compute the direvative of function PetscMax(0, 2*f*(1-f) / (PetscSqr(f) + PetscSqr(1-f)))*/
  PetscScalar result = 0.0;
  if (f*(1-f)<0.0) {
    result = 0.0;
  }else{
    result = (2-4*f)/(PetscSqr(f) + PetscSqr(1-f)) + (2*f*(1-f)*(4*f-2))/PetscSqr(PetscSqr(f) + PetscSqr(1-f));
  }

  return result;
}

/************************Limiter Sin*****************************************/
PetscReal Limit_Sin(PetscReal f)
{
  PetscReal fclip = PetscMax(0,PetscMin(f,1));
  return PetscSinReal(PETSC_PI*fclip);
}
PetscReal Limit_SinGrad(PetscReal f)
{/*Compute the direvative of function PetscSinReal(PETSC_PI*PetscMax(0,PetscMin(f,1)))*/
  PetscScalar result = 0.0;
  if (f<1.0 && f>0.0) {
    result = PETSC_PI*PetscCosReal(PETSC_PI*f);
  }else{
    result = 0.0;
  }
  return result;
}

/************************Limiter Superbee*************************************/
PetscReal Limit_Superbee(PetscReal f)
{
  return 2*Limit_Minmod(f);
}
PetscReal Limit_SuperbeeGrad(PetscReal f)
{/*Compute the direvative of function Limit_Superbee*/
  return 2*Limit_MinmodGrad(f);
}

/************************Limiter MC*****************************************/
PetscReal Limit_MC(PetscReal f)
{
  return PetscMin(1,Limit_Superbee(f));
}
PetscReal Limit_MCGrad(PetscReal f)
{/*Compute the direvative of function Limit_MC*/
  PetscScalar result = 0.0;
  if (Limit_Superbee(f)<1.0){
    result = Limit_SuperbeeGrad(f);
  }else{
    result = 0.0;
  }
  return result;
}


PetscScalar DotDIM(const PetscScalar *x,const PetscScalar *y)
{
  PetscInt    i;
  PetscScalar prod=0.0;

  for (i=0; i<DIM; i++) prod += x[i]*y[i];
  return prod;
}

PetscReal NormDIM(const PetscScalar *x)
{
  return PetscSqrtReal(PetscAbsScalar(DotDIM(x,x)));
}

void axDIM(const PetscScalar a,PetscScalar *x)
{
  PetscInt i;
  for (i=0; i<DIM; i++) x[i] *= a;
}

void waxDIM(const PetscScalar a,const PetscScalar *x, PetscScalar *w)
{
  PetscInt i;
  for (i=0; i<DIM; i++) w[i] = x[i]*a;
}
void NormalSplitDIM(const PetscReal *n,const PetscScalar *x,PetscScalar *xn,PetscScalar *xt)
{                               /* Split x into normal and tangential components */
  PetscInt    i;
  PetscScalar c;
  c = DotDIM(x,n)/DotDIM(n,n);
  for (i=0; i<DIM; i++) {
    xn[i] = c*n[i];
    xt[i] = x[i]-xn[i];
  }
}

PetscScalar Dot2(const PetscScalar *x,const PetscScalar *y)
{
  return x[0]*y[0] + x[1]*y[1];
}

PetscReal Norm2(const PetscScalar *x)
{
  return PetscSqrtReal(PetscAbsScalar(Dot2(x,x)));
}

void Normalize2(PetscScalar *x)
{
  PetscReal a = 1./Norm2(x); x[0] *= a; x[1] *= a;
}

void Waxpy2(PetscScalar a,const PetscScalar *x,const PetscScalar *y,PetscScalar *w)
{
  w[0] = a*x[0] + y[0]; w[1] = a*x[1] + y[1];
}

void Scale2(PetscScalar a,const PetscScalar *x,PetscScalar *y)
{
  y[0] = a*x[0]; y[1] = a*x[1];
}

void WaxpyD(PetscInt dim, PetscScalar a, const PetscScalar *x, const PetscScalar *y, PetscScalar *w)
{
  PetscInt d; for (d = 0; d < dim; ++d) w[d] = a*x[d] + y[d];
}

PetscScalar DotD(PetscInt dim, const PetscScalar *x, const PetscScalar *y)
{
  PetscScalar sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += x[d]*y[d]; return sum;
}
PetscReal NormD(PetscInt dim, const PetscScalar *x)
{
  return PetscSqrtReal(PetscAbsScalar(DotD(dim,x,x)));
}

void NormalSplit(const PetscReal *n,const PetscScalar *x,PetscScalar *xn,PetscScalar *xt)
{                               /* Split x into normal and tangential components */
  Scale2(Dot2(x,n)/Dot2(n,n),n,xn);
  Waxpy2(-1,xn,x,xt);
}

PetscReal SourceRho(User user, const PetscScalar *cgrad, const PetscScalar *x, const PetscReal *xcoord)
{
  return 0.0;
}

#undef __FUNCT__
#define __FUNCT__ "SourceU"
/*@C
  This function is for the terms:
  gradp = - R \frac{\partial \rho T}{x}
  and
   f_u

  Input Parameters:
+ user  - The user defined contex
. cgrad - The gradients at the centoid of the element
- x - The value of the variables at the centeroid of the cell
+ xcoord - The coordinates of the cell centriod

@*/
PetscReal SourceU(User user, const PetscScalar *cgrad, const PetscScalar *x, const PetscReal *xcoord)
{
  PetscReal       gradu[DIM][DIM];
  PetscReal       gradRhoU[DIM][DIM];
  PetscReal       gradRho[DIM];
  PetscReal       gradRhoE[DIM];
  PetscInt        i, j;
  PetscReal       f, result;
  const Node      *X = (const Node*)x;
  PetscReal       u[DIM];
  PetscReal       Temp;

  f = 0.0;
  if(user->benchmark_couette) {
    PetscScalar U, H, T;

    U = 0.3; H = 10;
    T = user->T0 + xcoord[1]/H*(user->T1 - user->T0) + user->viscosity*U*U/(2*user->k)*xcoord[1]/H*(1.0 - xcoord[1]/H);
    f = (1/T)*(U/H)*xcoord[1];
    result = f;
  }

  if (user->PressureFlux){
    result = f;
  }else{
    for (i=0; i<DIM; i++) {
      u[i] = X->ru[i]/X->r;
    }

    for (i=0; i<DIM; i++) {
      gradRho[i] = cgrad[i];
    }/*gradient of the density*/

    for (i=0; i<DIM; i++) {
      for (j=0; j<DIM; j++) {
        gradRhoU[i][j] = cgrad[(i+1)*DIM + j];
      }/*gradient of the density velocity*/
    }

    for (i=0; i<DIM; i++) {
      gradRhoE[i] = cgrad[4*DIM + i];
    }/*gradient of the energy: \nabla (rho*E)*/

    for (i=0; i<DIM; i++) {
      for (j=0; j<DIM; j++) {
        gradu[i][j] = (gradRhoU[i][j] - gradRho[j]*u[i])/X->r;
      }/*gradient of the velocity*/
    }

    Temp = gradRhoU[0][0]*u[0]  + gradRhoU[1][0]*u[1]  + gradRhoU[2][0]*u[2]
         + gradu[0][0]*X->ru[0] + gradu[1][0]*X->ru[1] + gradu[2][0]*X->ru[2]; //\nabla_x(rho*u*u)

    result = (user->adiabatic-1)*(gradRhoE[0] - 0.5*Temp) + f;

    //result = -user->R*(gradRho[0]*X->rE + gradRhoE[0]*X->r) + f;
    //PetscPrintf(PETSC_COMM_WORLD, "result = %f\n", result);
  }

  return result;
}

#undef __FUNCT__
#define __FUNCT__ "SourceV"
/*@C
  This function is for the terms:
  gradp = - R \frac{\partial \rho T}{y}
  and
   f_v

  Input Parameters:
+ user  - The user defined contex
. cgrad - The gradients at the centoid of the element
- x - The value of the variables at the centeroid of the cell
+ xcoord - The coordinates of the cell centriod

@*/
PetscReal SourceV(User user, const PetscScalar *cgrad, const PetscScalar *x, const PetscReal *xcoord)
{
PetscReal       gradu[DIM][DIM];
  PetscReal       gradRhoU[DIM][DIM];
  PetscReal       gradRho[DIM];
  PetscReal       gradRhoE[DIM];
  PetscInt        i, j;
  PetscReal       f, result;
  const Node      *X = (const Node*)x;
  PetscReal       u[DIM];
  PetscReal       Temp;

  f = 0.0;
  if(user->benchmark_couette) {
    PetscScalar U, H, T;

    U = 0.3; H = 10;
    T = user->T0 + xcoord[1]/H*(user->T1 - user->T0) + user->viscosity*U*U/(2*user->k)*xcoord[1]/H*(1.0 - xcoord[1]/H);
    f = (1/T)*(U/H)*xcoord[1];
    result = f;
  }

  if (user->PressureFlux){
    result = f;
  }else{
    for (i=0; i<DIM; i++) {
      u[i] = X->ru[i]/X->r;
    }

    for (i=0; i<DIM; i++) {
      gradRho[i] = cgrad[i];
    }/*gradient of the density*/

    for (i=0; i<DIM; i++) {
      for (j=0; j<DIM; j++) {
        gradRhoU[i][j] = cgrad[(i+1)*DIM + j];
      }/*gradient of the density velocity*/
    }

    for (i=0; i<DIM; i++) {
      gradRhoE[i] = cgrad[4*DIM + i];
    }/*gradient of the energy: \nabla (rho*E)*/

    for (i=0; i<DIM; i++) {
      for (j=0; j<DIM; j++) {
        gradu[i][j] = (gradRhoU[i][j] - gradRho[j]*u[i])/X->r;
      }/*gradient of the velocity*/
    }

    Temp = gradRhoU[0][1]*u[0]  + gradRhoU[1][1]*u[1]  + gradRhoU[2][1]*u[2]
         + gradu[0][1]*X->ru[0] + gradu[1][1]*X->ru[1] + gradu[2][1]*X->ru[2]; //\nabla_x(rho*u*u)

    result = (user->adiabatic-1)*(gradRhoE[1] - 0.5*Temp) + f;

    //result = -user->R*(gradRho[1]*X->rE + gradRhoE[1]*X->r) + f;
    //PetscPrintf(PETSC_COMM_WORLD, "result = %f\n", result);
  }

  return result;
}

#undef __FUNCT__
#define __FUNCT__ "SourceW"
/*@C
  This function is for the terms:
  gradp = - R \frac{\partial \rho T}{z}
  and
   f_w

  Input Parameters:
+ user  - The user defined contex
. cgrad - The gradients at the centoid of the element
- x - The value of the variables at the centeroid of the cell
+ xcoord - The coordinates of the cell centriod

@*/
PetscReal SourceW(User user, const PetscScalar *cgrad, const PetscScalar *x, const PetscReal *xcoord)
{
  PetscReal       gradu[DIM][DIM];
  PetscReal       gradRhoU[DIM][DIM];
  PetscReal       gradRho[DIM];
  PetscReal       gradRhoE[DIM];
  PetscInt        i, j;
  PetscReal       f, result;
  const Node      *X = (const Node*)x;
  PetscReal       u[DIM];
  PetscReal       Temp;

  f = 0.0;
  if(user->benchmark_couette) {
    PetscScalar U, H, T;

    U = 0.3; H = 10;
    T = user->T0 + xcoord[1]/H*(user->T1 - user->T0) + user->viscosity*U*U/(2*user->k)*xcoord[1]/H*(1.0 - xcoord[1]/H);
    f = (1/T)*(U/H)*xcoord[1];
    result = f;
  }

  if (user->PressureFlux){
    result = f;
  }else{
    for (i=0; i<DIM; i++) {
      u[i] = X->ru[i]/X->r;
    }

    for (i=0; i<DIM; i++) {
      gradRho[i] = cgrad[i];
    }/*gradient of the density*/

    for (i=0; i<DIM; i++) {
      for (j=0; j<DIM; j++) {
        gradRhoU[i][j] = cgrad[(i+1)*DIM + j];
      }/*gradient of the density velocity*/
    }

    for (i=0; i<DIM; i++) {
      gradRhoE[i] = cgrad[4*DIM + i];
    }/*gradient of the energy: \nabla (rho*E)*/

    for (i=0; i<DIM; i++) {
      for (j=0; j<DIM; j++) {
        gradu[i][j] = (gradRhoU[i][j] - gradRho[j]*u[i])/X->r;
      }/*gradient of the velocity*/
    }

    Temp = gradRhoU[0][2]*u[0]  + gradRhoU[1][2]*u[1]  + gradRhoU[2][2]*u[2]
         + gradu[0][2]*X->ru[0] + gradu[1][2]*X->ru[1] + gradu[2][2]*X->ru[2]; //\nabla_x(rho*u*u)

    result = (user->adiabatic-1)*(gradRhoE[2] - 0.5*Temp) + f;

    //result = -user->R*(gradRho[2]*X->rE + gradRhoE[2]*X->r) + f;
    //PetscPrintf(PETSC_COMM_WORLD, "result = %f\n", result);
  }

  return result;
}

#undef __FUNCT__
#define __FUNCT__ "SourceE"
/*@C
  This function is for the terms:
  \Phi = \mu*(2*((u_x)^2 + (u_y)^2 + (u_z)^2) + (u_y + v_x)^2 + (u_z + w_x)^2 + (v_z + w_y)^2),
  and
  Other = R*(u, v, w)\cdot grad(\rho*T)
        = R*(u*(\rho*T_x + T*\rho_x) + v*(\rho*T_y + T*\rho_y) + w*(\rho*T_z + T*\rho_z))
  and
   f_e

  Input Parameters:
+ user  - The user defined contex
. cgrad - The gradients at the centoid of the element
- x - The value of the variables at the centeroid of the cell
+ xcoord - The coordinates of the cell centriod

@*/
PetscReal SourceE(User user, const PetscScalar *cgrad, const PetscScalar *x, const PetscReal *xcoord)
{
  PetscReal       gradu[DIM][DIM];
  PetscReal       gradRho[DIM];
  PetscReal       gradT[DIM];
  PetscInt        i, j;
  PetscReal       f, phi, other, result;
  const Node      *X = (const Node*)x;

  f = 0.0;
  for (i=0; i<DIM; i++) {
    gradRho[i] = cgrad[i];
  }/*gradient of the density*/

  for (i=0; i<DIM; i++) {
    for (j=0; j<DIM; j++) {
      gradu[i][j] = cgrad[(i+1)*DIM + j];
    }/*gradient of the velocity*/
  }

  for (i=0; i<DIM; i++) {
    gradT[i] = cgrad[4*DIM + i];
  }/*gradient of the energy*/

  phi = user->viscosity*(2*(gradu[0][0]*gradu[0][0] + gradu[1][1]*gradu[1][1] + gradu[2][2]*gradu[2][2])
                        + (gradu[0][1] + gradu[1][0])*(gradu[0][1] + gradu[1][0])
                        + (gradu[0][2] + gradu[2][0])*(gradu[0][2] + gradu[2][0])
                        + (gradu[1][2] + gradu[2][1])*(gradu[1][2] + gradu[2][1]));

  other = user->R*(  X->ru[0]*(X->r*gradT[0] + X->rE*gradRho[0])
                   + X->ru[1]*(X->r*gradT[1] + X->rE*gradRho[1])
                   + X->ru[2]*(X->r*gradT[2] + X->rE*gradRho[2]));
  result = phi + other + f;
  if(user->Euler) result = f;

  return result;
}

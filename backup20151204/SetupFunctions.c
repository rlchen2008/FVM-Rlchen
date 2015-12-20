#include <petscts.h>
#include <petscfv.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <petscblaslapack.h>
#include "AeroSim.h"

#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <../src/sys/utils/hash.h>
#include <petsc-private/isimpl.h>

PETSC_EXTERN PetscErrorCode indicesPointFields_private(PetscSection section, PetscInt point, PetscInt off,
                                         PetscInt foffs[], PetscBool setBC, PetscInt orientation, PetscInt indices[]);
PETSC_EXTERN PetscErrorCode indicesPoint_private(PetscSection section, PetscInt point, PetscInt off,
                                         PetscInt *loff, PetscBool setBC, PetscInt orientation, PetscInt indices[]);


#undef __FUNCT__
#define __FUNCT__ "FormFunction"
/**
   Form the nonlinear function for the SNES
   @param, vec y: input vector, the initial guess of solution
           vec out: output vector.
**/
PetscErrorCode FormFunction(SNES snes, Vec y, Vec out, void *ctx)
{
  User user = (User) ctx;
  Algebra algebra = user->algebra;

  PetscErrorCode ierr;
  PetscMPIInt rank;


  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* preset the out vector y */
  ierr = VecSet(out, 0.0);CHKERRQ(ierr);

  /* steady-state */
  if(user->timestep == TIMESTEP_STEADY_STATE){

    /* form the steady state nonlinear function */
    ierr = FormTimeStepFunction(user, algebra, y, algebra->fn);CHKERRQ(ierr);
 //   ierr = VecView(algebra->fn, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = VecAXPY(out, 1.0, algebra->fn);CHKERRQ(ierr);
//    ierr = VecView(out, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  }else if(user->timestep == TIMESTEP_BEULER){ /* backward euler timestep */

    ierr = FormMassTimeStepFunction(user, algebra, y,
				    algebra->fn,PETSC_TRUE);CHKERRQ(ierr);
    ierr = VecAXPY(out, 1.0, algebra->fn);CHKERRQ(ierr);
    ierr = FormMassTimeStepFunction(user, algebra, algebra->oldsolution,
				    algebra->fn,PETSC_FALSE);CHKERRQ(ierr);
    ierr = VecAXPY(out, -1.0, algebra->fn);CHKERRQ(ierr);

    ierr = FormTimeStepFunction(user, algebra, y, algebra->fn);CHKERRQ(ierr);
    ierr = VecAXPY(out, user->dt, algebra->fn);CHKERRQ(ierr);

  }else if(user->timestep == TIMESTEP_BDF){
    ierr = FormMassTimeStepFunction(user, algebra, y,
				    algebra->fn,PETSC_TRUE);CHKERRQ(ierr);
    ierr = VecAXPY(out, 1.5, algebra->fn);CHKERRQ(ierr);
    ierr = FormMassTimeStepFunction(user, algebra, algebra->oldsolution,
				    algebra->fn,PETSC_FALSE);CHKERRQ(ierr);
    ierr = VecAXPY(out, -2.0, algebra->fn);CHKERRQ(ierr);
    ierr = FormMassTimeStepFunction(user, algebra, algebra->oldoldsolution,
				    algebra->fn,PETSC_FALSE);CHKERRQ(ierr);
    ierr = VecAXPY(out, 0.5, algebra->fn);CHKERRQ(ierr);

    ierr = FormTimeStepFunction(user, algebra, y, algebra->fn);CHKERRQ(ierr);
    ierr = VecAXPY(out, user->dt, algebra->fn);CHKERRQ(ierr);
  }else if(user->timestep == TIMESTEP_TRAPEZOIDAL){
    ierr = FormMassTimeStepFunction(user, algebra, y,
				    algebra->fn,PETSC_TRUE);CHKERRQ(ierr);
    ierr = VecAXPY(out, 1.0, algebra->fn);CHKERRQ(ierr);
    ierr = FormMassTimeStepFunction(user, algebra, algebra->oldsolution,
				    algebra->fn,PETSC_FALSE);CHKERRQ(ierr);
    ierr = VecAXPY(out, -1.0, algebra->fn);CHKERRQ(ierr);

    ierr = VecAXPY(out, 0.5 * user->dt, algebra->oldfn);CHKERRQ(ierr);
    ierr = FormTimeStepFunction(user, algebra, y, algebra->fn);CHKERRQ(ierr);
    ierr = VecAXPY(out, 0.5 * user->dt, algebra->fn);CHKERRQ(ierr);

  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormMassTimeStepFunction"
/**
   Form the time step function for the mass term
   @param Vec in:       (y^(n+1))
   out @param Vec out:  My^(n+1) - M y^(n)    if timestep = 1, backward euler
                                              will do the BDF2
**/
PetscErrorCode FormMassTimeStepFunction(User user, Algebra algebra, Vec in, Vec out, PetscBool rebuild)
{
  PetscErrorCode  ierr;
  PetscMPIInt     rank;
  Vec            inLocal;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = VecSet(out, 0.0);CHKERRQ(ierr);

  ierr = DMGetLocalVector(user->dm, &inLocal);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(user->dm, in, INSERT_VALUES, inLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dm, in, INSERT_VALUES, inLocal);CHKERRQ(ierr);

  ierr = ApplyBC(user->dm, user->current_time, inLocal, user);CHKERRQ(ierr);

  ierr = CaculateLocalMassFunction(user->dm, inLocal, out, user);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(user->dm, inLocal, INSERT_VALUES, in);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->dm, inLocal, INSERT_VALUES, in);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(user->dm, &inLocal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormTimeStepFunction"
/**
   Creates \f$ f_n \f$ part of time-stepping scheme.

   For ODE solvers (aka time-stepping schemes), you think of your problem as \f$ y' = f(y) \f$ and

   Note that this actually returns something more like \f$ -f_n \f$.
 */
PetscErrorCode FormTimeStepFunction(User user, Algebra algebra, Vec in, Vec out)
{
  PetscErrorCode  ierr;
  PetscMPIInt     rank;
  Vec             inLocal;
  DM              dmFace, dmCell;


  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = VecSet(out, 0.0);CHKERRQ(ierr);
//  ierr = VecView(in, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  /*Since the DMPlexVecSetClosure only works on the local vectors,
    we need to create a local vector and scatter the global
    vector to the local vector and insert the values,
    and then scatter the local updated vectors back to the global vector.*/
  ierr = DMGetLocalVector(user->dm, &inLocal);CHKERRQ(ierr);
  ierr = VecSet(inLocal, 0);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(user->dm, in, INSERT_VALUES, inLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dm, in, INSERT_VALUES, inLocal);CHKERRQ(ierr);

  ierr = ApplyBC(user->dm, user->current_time, inLocal, user);CHKERRQ(ierr);

  ierr = VecGetDM(user->facegeom,&dmFace);CHKERRQ(ierr);
  ierr = VecGetDM(user->cellgeom,&dmCell);CHKERRQ(ierr);

  ierr = ConstructCellCentriodGradient(user->dm, dmFace, dmCell, user->current_time, inLocal, out, user);CHKERRQ(ierr);
  /*Construct the cell gradient at the current time
    and save it into the user->dmGrad. If you want
    to use the cell gradient, following these:
    ierr = DMGetGlobalVector(user->dmGrad,&Grad);CHKERRQ(ierr); or
    ierr = DMGetLocalVector(user->dmGrad,&Grad);CHKERRQ(ierr);
  */
  //VecView(out,PETSC_VIEWER_STDOUT_WORLD);
  if (user->second_order){
    ierr = CaculateLocalFunction_LS(user->dm, dmFace, dmCell, user->current_time, inLocal, out, user);CHKERRQ(ierr);
  }else{
    ierr = CaculateLocalFunction_Upwind(user->dm, dmFace, dmCell, user->current_time, inLocal, out, user);CHKERRQ(ierr);
  }

  //VecView(out,PETSC_VIEWER_STDOUT_WORLD);
  ierr = CaculateLocalSourceTerm(user->dm, inLocal, out, user);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(user->dm, inLocal, INSERT_VALUES, in);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->dm, inLocal, INSERT_VALUES, in);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(user->dm, &inLocal);CHKERRQ(ierr);
  //VecView(in,PETSC_VIEWER_STDOUT_WORLD);
  VecView(out,PETSC_VIEWER_STDOUT_WORLD);
  if (1){
    PetscViewer    viewer;
    PetscReal fnnorm;

    ierr = VecNorm(out,NORM_INFINITY,&fnnorm);CHKERRQ(ierr);
    ierr = OutputVTK(user->dm, "function.vtk", &viewer);CHKERRQ(ierr);
    ierr = VecView(out, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Step %D at time %g with founction norm = %g \n",
                       user->current_step, user->current_time, fnnorm);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyRHSFunction"
/**
This function is for the explicit method, that is the right hand side
of the system: du/dt = F(u)
*/
PetscErrorCode MyRHSFunction(TS ts,PetscReal time,Vec in,Vec out,void *ctx)
{
  User           user = (User)ctx;
  DM             dm, dmFace, dmCell;
  PetscSection   section;
  Vec            inLocal;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  {
    PetscReal norm;
    PetscInt size;
    ierr = VecNorm(in,NORM_INFINITY,&norm);CHKERRQ(ierr);
    ierr = VecGetSize(in, &size);CHKERRQ(ierr);
    norm = norm/size;
    if (norm>1.e5) {
      SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_LIB,
      "The norm of the solution is: %f (current time: %f). The explicit method is going to DIVERGE!!!", norm, time);
    }
  }
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = VecGetDM(user->facegeom,&dmFace);CHKERRQ(ierr);
  ierr = VecGetDM(user->cellgeom,&dmCell);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&inLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, in, INSERT_VALUES, inLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, in, INSERT_VALUES, inLocal);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);

  ierr = ApplyBC(dm, time, inLocal, user);CHKERRQ(ierr);

  ierr = ConstructCellCentriodGradient(user->dm, dmFace, dmCell, time, inLocal, out, user);CHKERRQ(ierr);

  ierr = VecZeroEntries(out);CHKERRQ(ierr);
  if (user->second_order){
    ierr = CaculateLocalFunction_LS(user->dm, dmFace, dmCell, time, inLocal, out, user);CHKERRQ(ierr);
  }else{
    ierr = CaculateLocalFunction_Upwind(user->dm, dmFace, dmCell, time, inLocal, out, user);CHKERRQ(ierr);
  }
  ierr = CaculateLocalSourceTerm(user->dm, inLocal, out, user);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(user->dm, inLocal, INSERT_VALUES, in);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->dm, inLocal, INSERT_VALUES, in);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&inLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CaculateLocalMassFunction"
/*
 Use the first order upwind scheme to compute the flux
*/
PetscErrorCode CaculateLocalMassFunction(DM dm, Vec locX, Vec F, User user)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       *f;
  PetscInt          cStart, cell;
//  Physics           phys = user->model->physics;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);

  for (cell = cStart; cell < user->cEndInterior; cell++) {
    PetscInt          i;
    PetscScalar       *fref;
    const PetscScalar *xref;

    ierr = DMPlexPointGlobalRead(dm,cell,x,&xref);CHKERRQ(ierr); /*For the unkown variables*/
    ierr = DMPlexPointGlobalRef(dm,cell,f,&fref);CHKERRQ(ierr);
//    if (!fref){ PetscPrintf(PETSC_COMM_WORLD,"%d, %d\n", cell, user->cEndInterior);}
    if (fref){
      fref[0] = xref[0];/*the density*/
      for (i=1; i<DIM+1; i++) {
        fref[i] = xref[i];
      }/*viscosity*/
      fref[DIM+1] = xref[DIM+1];/*Energy*/
    }
  }

  ierr = VecRestoreArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CaculateLocalSourceTerm"
/*
 Caculate the source term of the equations, which includes all
 terms except convetion term, diffusion term, and timedependent term.
*/
PetscErrorCode CaculateLocalSourceTerm(DM dm, Vec locX, Vec F, User user)
{
  PetscErrorCode    ierr;
  DM                dmGrad = user->dmGrad;
  const PetscScalar *x;
  PetscScalar       *f;
  PetscInt          cStart, cell;
  const PetscScalar *cellgeom;
  const CellGeom    *cg;
  Vec               locGrad, Grad;
  const PetscScalar *grad;
  DM                dmCell;

  PetscFunctionBeginUser;

  ierr = VecGetDM(user->cellgeom,&dmCell);CHKERRQ(ierr);

  ierr = VecGetArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmGrad,&locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmGrad,Grad,INSERT_VALUES,locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmGrad,Grad,INSERT_VALUES,locGrad);CHKERRQ(ierr);

  ierr = DMRestoreGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);

  ierr = VecGetArrayRead(locGrad,&grad);CHKERRQ(ierr);

  for (cell = cStart; cell < user->cEndInterior; cell++) {
    PetscScalar       *fref;
    const PetscScalar *xref;
    PetscScalar       *cgrad;

    ierr = DMPlexPointLocalRead(dmCell,cell,cellgeom,&cg);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm,cell,x,&xref);CHKERRQ(ierr); /*For the unkown variables*/
    ierr = DMPlexPointGlobalRef(dm,cell,f,&fref);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmGrad,cell,grad,&cgrad);CHKERRQ(ierr);
//    if (!fref){ PetscPrintf(PETSC_COMM_WORLD,"%d, %d\n", cell, user->cEndInterior);}
    if (fref){
      fref[0] += SourceRho(user, cgrad, xref, cg->centroid);/*the continuity equation*/
      fref[1] += SourceU(user, cgrad, xref, cg->centroid); /*Momentum U*/
      fref[2] += SourceV(user, cgrad, xref, cg->centroid); /*Momentum V*/
      fref[3] += SourceW(user, cgrad, xref, cg->centroid); /*Momentum W*/
      fref[4] += SourceE(user, cgrad, xref, cg->centroid);/*Energy*/
    }
  }

  ierr = VecRestoreArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(locGrad,&grad);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmGrad,&locGrad);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CaculateLocalFunction_Upwind"
/*
 Use the first order upwind scheme to compute the flux
*/
PetscErrorCode CaculateLocalFunction_Upwind(DM dm,DM dmFace,DM dmCell,PetscReal time,Vec locX,Vec F,User user)
{
  Physics           phys = user->model->physics;
  DM                dmGrad = user->dmGrad;
  PetscErrorCode    ierr;
  const PetscScalar *facegeom, *cellgeom, *x;
  PetscScalar       *f;
  PetscInt          fStart, fEnd, face;

  Vec               locGrad, Grad;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(user->facegeom, &facegeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->cellgeom, &cellgeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX, &x);CHKERRQ(ierr);
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dmGrad, &Grad);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmGrad, &locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmGrad, Grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmGrad, Grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);

  ierr = DMRestoreGlobalVector(dmGrad, &Grad);CHKERRQ(ierr);

  const PetscScalar *grad;
  ierr = VecGetArrayRead(locGrad, &grad);CHKERRQ(ierr);

  {
    const PetscInt    *cells;
    PetscInt          i,ghost;
    PetscScalar       *fluxcon, *fluxdiff, *fL,*fR;
    const FaceGeom    *fg;
    const CellGeom    *cgL,*cgR;
    const PetscScalar *xL,*xR;
    const PetscScalar *cgrad[2];
    PetscReal         FaceArea;

    ierr = PetscMalloc(phys->dof * sizeof(PetscScalar), &fluxcon);CHKERRQ(ierr); /*For the convection terms*/
    ierr = PetscMalloc(phys->dof * sizeof(PetscScalar), &fluxdiff);CHKERRQ(ierr); /*For the diffusion terms*/

    for (face = fStart; face < fEnd; ++face) {
      ierr = DMPlexGetLabelValue(dm, "ghost", face, &ghost);CHKERRQ(ierr);
      if (ghost >= 0) continue;
      ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);/*The support of a face is the cells (two cells)*/
      ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);/*Read the data from "facegeom" for the point "face"*/
      ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR);CHKERRQ(ierr);
/*
    PetscPrintf(PETSC_COMM_SELF, "face[%d]:(%f, %f, %f), cgL:(%f, %f, %f), cgR:(%f, %f, %f), fnorm:(%f, %f, %f)\n",
                 face, fg->centroid[0], fg->centroid[1], fg->centroid[2],
                 cgL->centroid[0], cgL->centroid[1], cgL->centroid[2], cgR->centroid[0], cgR->centroid[1], cgR->centroid[2],
                 fg->normal[0], fg->normal[1], fg->normal[2]);
*/
      ierr = DMPlexPointLocalRead(dm, cells[0], x, &xL);CHKERRQ(ierr); /*For the unkown variables*/
      ierr = DMPlexPointLocalRead(dm, cells[1], x, &xR);CHKERRQ(ierr);
      ierr = DMPlexPointGlobalRef(dm, cells[0], f, &fL);CHKERRQ(ierr); /*For the functions*/
      ierr = DMPlexPointGlobalRef(dm, cells[1], f, &fR);CHKERRQ(ierr);

      ierr = DMPlexPointLocalRead(dmGrad, cells[0], grad, &cgrad[0]);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmGrad, cells[1], grad, &cgrad[1]);CHKERRQ(ierr);

      ierr = RiemannSolver(user, cgrad[0], cgrad[1], fg->centroid, cgL->centroid, cgR->centroid, fg->normal, xL, xR, fluxcon, fluxdiff);CHKERRQ(ierr);
    /*Caculate the flux*/
      ierr = DMPlexComputeCellGeometryFVM(dm, face, &FaceArea, NULL, NULL);CHKERRQ(ierr);
      //PetscPrintf(PETSC_COMM_SELF, "FaceArea=%f, Volume=%f\n",FaceArea,cgL->volume);
    /*Compute the face area*/
   // FaceArea = 0.0;
      for (i=0; i<phys->dof; i++) {
/*
      PetscPrintf(PETSC_COMM_SELF, "face[%d]:(%f, %f, %f), GradL[%d] = (%f, %f, %f), GradR[%d] = (%f, %f, %f), fluxcon[%d] = %f\n",
                                    face, fg->centroid[0], fg->centroid[1], fg->centroid[2],
                                    i, cgrad[0][DIM*i], cgrad[0][DIM*i + 1], cgrad[0][DIM*i + 2],
                                    i, cgrad[1][DIM*i], cgrad[1][DIM*i + 1], cgrad[1][DIM*i + 2], i, fluxcon[i]);
*/

        if (fL) {
          fL[i] -= FaceArea*(fluxcon[i] + fluxdiff[i])/cgL->volume;
          PetscPrintf(PETSC_COMM_SELF, "cell[0] (%3.4g, %3.4g, %3.4g), face (%3.4g, %3.4g, %3.4g),fluxcon[i]=%3.4g, FaceArea=%3.4g, volume=%3.4g, fL[%d]= %3.4g\n",cgL->centroid[0], cgL->centroid[1], cgL->centroid[2],fg->centroid[0], fg->centroid[1], fg->centroid[2], fluxcon[i], FaceArea, cgL->volume, i, fL[i]);
        }
        if (fR) {
          fR[i] += FaceArea*(fluxcon[i] + fluxdiff[i])/cgR->volume;
          PetscPrintf(PETSC_COMM_SELF, "cell[0] (%3.4g, %3.4g, %3.4g), face (%3.4g, %3.4g, %3.4g),fluxcon[i]=%3.4g, FaceArea=%3.4g, volume=%3.4g, fR[%d]= %3.4g\n",cgL->centroid[0], cgL->centroid[1], cgL->centroid[2],fg->centroid[0], fg->centroid[1], fg->centroid[2], fluxcon[i], FaceArea, cgL->volume, i, fR[i]);
        }

      }

      printf("\n");
    }
    ierr = PetscFree(fluxcon);CHKERRQ(ierr);
    ierr = PetscFree(fluxdiff);CHKERRQ(ierr);
  }
//  VecView(F,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecRestoreArrayRead(locGrad,&grad);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmGrad,&locGrad);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CaculateLocalFunction_LS"
/**
  Caculate the flux using the Monotonic Upstream-Centered Scheme for Conservation Laws (van Leer, 1979)
*/
PetscErrorCode CaculateLocalFunction_LS(DM dm,DM dmFace,DM dmCell,PetscReal time,Vec locX,Vec F,User user)
{
  DM                dmGrad = user->dmGrad;
  Model             mod    = user->model;
  Physics           phys   = mod->physics;
  const PetscInt    dof    = phys->dof;
  PetscErrorCode    ierr;
  const PetscScalar *facegeom, *cellgeom, *x;
  PetscScalar       *f;
  PetscInt          fStart, fEnd, face, cStart, cell;
  Vec               locGrad, locGradLimiter, Grad;
/*
    Here the localGradLimiter refers to the gradient that
    has been multiplied by the limiter function.
    The locGradLimiter is used to construct the uL and uR,
    and the locGrad is used to caculate the diffusion term
*/
  Vec               TempVec; /*a temperal vec for the vector restore*/

  PetscFunctionBeginUser;
  ierr = DMGetGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);
  ierr = VecDuplicate(Grad, &TempVec);CHKERRQ(ierr);
  ierr = VecCopy(Grad, TempVec);CHKERRQ(ierr);
/*
  Backup the original vector and use it to restore the value of dmGrad,
  because I do not want to change the values of the cell gradient.
*/
  ierr = VecGetArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);

  {
    PetscScalar *grad;
    ierr = VecGetArray(Grad,&grad);CHKERRQ(ierr);
    const PetscInt    *faces;
    PetscInt          numFaces,f;
    PetscReal         *cellPhi; // Scalar limiter applied to each component separately
    const PetscScalar *cx;
    const CellGeom    *cg;
    PetscScalar       *cgrad;
    PetscInt          i;

    ierr = PetscMalloc(phys->dof*sizeof(PetscScalar),&cellPhi);CHKERRQ(ierr);
    // Limit interior gradients. Using cell-based loop because it generalizes better to vector limiters.
    for (cell=cStart; cell<user->cEndInterior; cell++) {
      ierr = DMPlexGetConeSize(dm,cell,&numFaces);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm,cell,&faces);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dm,cell,x,&cx);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell,cell,cellgeom,&cg);CHKERRQ(ierr);
      ierr = DMPlexPointGlobalRef(dmGrad,cell,grad,&cgrad);CHKERRQ(ierr);
      if (!cgrad) continue;     // ghost cell, we don't compute
      // Limiter will be minimum value over all neighbors
      for (i=0; i<dof; i++) cellPhi[i] = PETSC_MAX_REAL;

      for (f=0; f<numFaces; f++) {
        const PetscScalar *ncx;
        const CellGeom    *ncg;
        const PetscInt    *fcells;
        PetscInt          face = faces[f],ncell;
        PetscScalar       v[DIM];
        PetscBool         ghost;
        ierr = IsExteriorGhostFace(dm,face,&ghost);CHKERRQ(ierr);
        if (ghost) continue;
        ierr  = DMPlexGetSupport(dm,face,&fcells);CHKERRQ(ierr);
        ncell = cell == fcells[0] ? fcells[1] : fcells[0];
        // The expression (x ? y : z) has the value of y if x is nonzero, z otherwise
        ierr  = DMPlexPointLocalRead(dm,ncell,x,&ncx);CHKERRQ(ierr);
        ierr  = DMPlexPointLocalRead(dmCell,ncell,cellgeom,&ncg);CHKERRQ(ierr);
        Waxpy2(-1, cg->centroid, ncg->centroid, v);
        for (i=0; i<dof; i++) {
          // We use the symmetric slope limited form of Berger, Aftosmis, and Murman 2005
          PetscScalar phi,flim = 0.5 * (ncx[i] - cx[i]) / Dot2(&cgrad[i*DIM],v);
          phi        = (*user->Limit)(flim);
          cellPhi[i] = PetscMin(cellPhi[i],phi);
        }
      }
      // Apply limiter to gradient
      for (i=0; i<dof; i++) Scale2(cellPhi[i],&cgrad[i*DIM],&cgrad[i*DIM]);
    }
    ierr = PetscFree(cellPhi);CHKERRQ(ierr);
    ierr = VecRestoreArray(Grad,&grad);CHKERRQ(ierr);
  }
  ierr = DMGetLocalVector(dmGrad,&locGradLimiter);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmGrad,Grad,INSERT_VALUES,locGradLimiter);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmGrad,Grad,INSERT_VALUES,locGradLimiter);CHKERRQ(ierr);

  ierr = VecCopy(TempVec, Grad);CHKERRQ(ierr);//Restore the vector

  ierr = DMGetLocalVector(dmGrad,&locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmGrad,Grad,INSERT_VALUES,locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmGrad,Grad,INSERT_VALUES,locGrad);CHKERRQ(ierr);

  ierr = DMRestoreGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);
  ierr = VecDestroy(&TempVec);CHKERRQ(ierr);

  {
    const PetscScalar *grad, *gradlimiter;
    const PetscInt    *cells;
    PetscInt          ghost,i,j;
    PetscScalar       *fluxcon, *fluxdiff, *fx[2],*cf[2];
    const FaceGeom    *fg;
    const CellGeom    *cg[2];
    const PetscScalar *cx[2],*cgrad[2], *cgradlimiter[2];
    PetscScalar       *uL, *uR;
    PetscReal         FaceArea;

    ierr = VecGetArrayRead(locGrad,&grad);CHKERRQ(ierr);
    ierr = VecGetArrayRead(locGradLimiter,&gradlimiter);CHKERRQ(ierr);
    ierr = VecGetArray(F,&f);CHKERRQ(ierr);

    ierr = PetscMalloc(phys->dof * sizeof(PetscScalar), &fluxcon);CHKERRQ(ierr); // For the convection terms
    ierr = PetscMalloc(phys->dof * sizeof(PetscScalar), &fluxdiff);CHKERRQ(ierr); // For the diffusion terms
    ierr = PetscMalloc(phys->dof * sizeof(PetscScalar), &uL);CHKERRQ(ierr);
    ierr = PetscMalloc(phys->dof * sizeof(PetscScalar), &uR);CHKERRQ(ierr);// Please do not put the Malloc function into a for loop!!!!

    for (face=fStart; face<fEnd; face++) {
      fx[0] = uL; fx[1] = uR;

      ierr = DMPlexGetLabelValue(dm, "ghost", face, &ghost);CHKERRQ(ierr);
      if (ghost >= 0) continue;
      ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmFace,face,facegeom,&fg);CHKERRQ(ierr);
      for (i=0; i<2; i++) {
        PetscScalar dx[DIM];
        ierr = DMPlexPointLocalRead(dmCell,cells[i],cellgeom,&cg[i]);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dm,cells[i],x,&cx[i]);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dmGrad,cells[i],gradlimiter,&cgradlimiter[i]);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dmGrad,cells[i],grad,&cgrad[i]);CHKERRQ(ierr);
        ierr = DMPlexPointGlobalRef(dm,cells[i],f,&cf[i]);CHKERRQ(ierr);
        Waxpy2(-1,cg[i]->centroid,fg->centroid,dx);
        for (j=0; j<dof; j++) {
          fx[i][j] = cx[i][j] + Dot2(cgradlimiter[i],dx);
        }
        // fx[0] and fx[1] are the value of the variables on the left and right
        //  side of the face, respectively, that is u_L and u_R.
      }

      ierr = RiemannSolver(user, cgrad[0], cgrad[1], fg->centroid, cg[0]->centroid, cg[1]->centroid, fg->normal, fx[0], fx[1], fluxcon, fluxdiff);CHKERRQ(ierr);

      ierr = DMPlexComputeCellGeometryFVM(dm, face, &FaceArea, NULL, NULL);CHKERRQ(ierr);
        // Compute the face area

      for (i=0; i<phys->dof; i++) {
        if (cf[0]){
          cf[0][i] -= FaceArea*(fluxcon[i] + fluxdiff[i])/cg[0]->volume;
        //  if (PetscAbsScalar(cf[0][i])<1.e-8) {
        //    cf[0][i] = 0.0;
        //  } // to avoid the too small number
        }
        if (cf[1]){
          cf[1][i] += FaceArea*(fluxcon[i] + fluxdiff[i])/cg[1]->volume;
         // if (PetscAbsScalar(cf[1][i])<1.e-8) {
         //   cf[1][i] = 0.0;
         // } // to avoid the too small number. is this necessary? not sure!
        }
       // The flux on the interface, for the cell[0], it is an outcoming flux and for the cell[1], it is
       //   an incoming flux.
      }
//      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(locGrad,&grad);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locGradLimiter,&gradlimiter);CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
    ierr = PetscFree(fluxcon);CHKERRQ(ierr);
    ierr = PetscFree(fluxdiff);CHKERRQ(ierr);
    ierr = PetscFree(uL);CHKERRQ(ierr);
    ierr = PetscFree(uR);CHKERRQ(ierr);
//    VecView(F,PETSC_VIEWER_STDOUT_WORLD);
  }
  ierr = VecRestoreArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmGrad,&locGradLimiter);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmGrad,&locGrad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "ConstructCellCentriodGradient"
/*
Compute the gradient of the variables at the center of the cell
by the least-square reconstruction method based on the function
BuildLeastSquares.
*/
PetscErrorCode ConstructCellCentriodGradient(DM dm,DM dmFace,DM dmCell,PetscReal time,Vec locX,Vec F,User user)
{
  DM                dmGrad = user->dmGrad;
  Model             mod    = user->model;
  Physics           phys   = mod->physics;
  const PetscInt    dof    = phys->dof;
  PetscErrorCode    ierr;
  const PetscScalar *facegeom, *cellgeom, *x;
  PetscInt          fStart, fEnd, face, cStart;
  Vec               Grad;

  PetscFunctionBeginUser;
  ierr = DMGetGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);
  ierr = VecZeroEntries(Grad);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  {
    PetscScalar *grad;
    ierr = VecGetArray(Grad,&grad);CHKERRQ(ierr);
    /* Reconstruct gradients */
    for (face=fStart; face<fEnd; face++) {
      const PetscInt    *cells;
      const PetscScalar *cx[2];
      const FaceGeom    *fg;
      PetscScalar       *cgrad[2];
      PetscInt          i,j;
      PetscBool         ghost;

      ierr = IsExteriorGhostFace(dm,face,&ghost);CHKERRQ(ierr);
      if (ghost) continue;
      ierr = DMPlexGetSupport(dm,face,&cells);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmFace,face,facegeom,&fg);CHKERRQ(ierr);
      for (i=0; i<2; i++) {
        ierr = DMPlexPointLocalRead(dm,cells[i],x,&cx[i]);CHKERRQ(ierr);
        ierr = DMPlexPointGlobalRef(dmGrad,cells[i],grad,&cgrad[i]);CHKERRQ(ierr);
      }
      for (i=0; i<dof; i++) {
        PetscScalar delta = cx[1][i] - cx[0][i];
        for (j=0; j<DIM; j++) {
          if (cgrad[0]) cgrad[0][i*DIM+j] += fg->grad[0][j] * delta;
          if (cgrad[1]) cgrad[1][i*DIM+j] -= fg->grad[1][j] * delta;
        }
      }
    }
    ierr = VecRestoreArray(Grad,&grad);CHKERRQ(ierr);
  }
  ierr = DMRestoreGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(locX,&x);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ConstructCellCentriodGradientJacobian"
/*
Compute the gradient of the variables at the center of the cell
by the least-square reconstruction method based on the function
BuildLeastSquares.
*/
PetscErrorCode ConstructCellCentriodGradientJacobian(DM dm,DM dmFace,DM dmCell,PetscReal time,Vec locX, PetscInt cell, PetscScalar CellValues[],User user)
{
  DM                dmGrad = user->dmGrad;
  Model             mod    = user->model;
  Physics           phys   = mod->physics;
  const PetscInt    dof    = phys->dof;
  PetscErrorCode    ierr;
  const PetscScalar *facegeom, *cellgeom, *x;
  PetscInt          fStart, fEnd, face, cStart;
  PetscInt          numFaces;
  const PetscInt    *faces;
  Vec               Grad;

  PetscFunctionBeginUser;

  ierr = DMGetGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);
  ierr = VecZeroEntries(Grad);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);

  ierr = DMPlexGetConeSize(dm,cell,&numFaces);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm,cell,&faces);CHKERRQ(ierr);

  {
    PetscScalar *grad;
    ierr = VecGetArray(Grad,&grad);CHKERRQ(ierr);
    /* Reconstruct gradients */
    for (face=0; face<numFaces; face++) {
      const PetscInt    *cells;
      const PetscScalar *cx[2];
      const FaceGeom    *fg;
      PetscScalar       *cgrad[2];
      PetscInt          i,j;
      PetscBool         ghost;

      ierr = IsExteriorGhostFace(dm,faces[face],&ghost);CHKERRQ(ierr);
      if (ghost) continue;
      ierr = DMPlexGetSupport(dm,faces[face],&cells);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmFace,faces[face],facegeom,&fg);CHKERRQ(ierr);
      for (i=0; i<2; i++) {
        ierr = DMPlexPointLocalRead(dm,cells[i],x,&cx[i]);CHKERRQ(ierr);
        ierr = DMPlexPointGlobalRef(dmGrad,cells[i],grad,&cgrad[i]);CHKERRQ(ierr);
      }
      for (i=0; i<dof; i++) {
        PetscScalar delta = cx[1][i] - cx[0][i];
        for (j=0; j<DIM; j++) {
          if (cgrad[0]) cgrad[0][i*DIM+j] += fg->grad[0][j] * delta;
          if (cgrad[1]) cgrad[1][i*DIM+j] -= fg->grad[1][j] * delta;
        }
      }
    }
    ierr = VecRestoreArray(Grad,&grad);CHKERRQ(ierr);
  }
  ierr = DMRestoreGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(locX,&x);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Energy"
/*
The internal energy per unit mass: e = E - 1/2*u*u
*/
PetscErrorCode Energy(User user,const Node *x,PetscScalar *e)
{
  PetscScalar  u[DIM]; // unscaled u
  PetscScalar  E; // unscaled total energy
  PetscInt     i;
  PetscScalar  uu; // u dot u

  PetscFunctionBeginUser;

  for (i=0; i<DIM; i++) {
    u[i] = x->ru[i]/x->r;
  }
  E = x->rE/x->r;

  uu = DotDIM(u, u);

  (*e) = E - 0.5*uu;

  //printf("E = %f, uu =%f, e = %f\n", E, uu, *e);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Pressure_Full"
/* For the equations that include energy
The state equations is: p = \rho RT, and e = p/(rho*(gamma-1))
where \rho is the density, T is the temperature,
and R is the gas constant which is 8.3144621 J K^{-1} mol^{-1}.

Note that here x->rE is the rho scaled total energy per unit mass r*E,
that is the total energy per unit volume
*/
PetscErrorCode Pressure_Full(User user,const Node *x,PetscScalar *p)
{
  PetscErrorCode ierr;
  PetscScalar e; // the internal energy per unit mass

  PetscFunctionBeginUser;

  ierr = Energy(user, x, &e); CHKERRQ(ierr);// Compute the iternal energy

  // p = rho*(gamma - 1)*e

  (*p) = x->r*(user->adiabatic-1)*e;

  if(user->benchmark_couette) { (*p) = 1.0/user->adiabatic;}

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Pressure_Partial"
/* For the equations that include energy
The state equations is: p = \rho RT,
where \rho is the density, T is the temperature,
and R is the gas constant which is 8.3144621 J K^{-1} mol^{-1}.

Note that here x->rE is the rho scaled total energy per unit mass r*E,
that is the total energy per unit volume
*/
PetscErrorCode Pressure_Partial(User user,const Node *x,PetscScalar *p)
{
  PetscErrorCode ierr;
  PetscScalar e; // the internal energy per unit mass

  PetscFunctionBeginUser;

  ierr = Energy(user, x, &e); CHKERRQ(ierr);// Compute the iternal energy

  // p = rho*(gamma - 1)*e

  (*p) = x->r*(user->R)*e;

  if(user->benchmark_couette) { (*p) = 1.0/user->adiabatic;}

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SpeedOfSound_PG"
/*
 For the ideal gas, the relationship of the speed of the sound "c",
 density "\rho" and the pressure "p" is:
 p = c^2 \rho
*/
PetscErrorCode SpeedOfSound_PG(User user,const Node *x,PetscScalar *c)
{
  PetscErrorCode ierr;
  PetscScalar p;

  PetscFunctionBeginUser;
  if (user->includeenergy){
    ierr = Pressure_Full(user,x,&p);CHKERRQ(ierr);
  }else{
    ierr = Pressure_Partial(user,x,&p);CHKERRQ(ierr);
  }

  (*c)=PetscSqrtScalar(user->adiabatic*PetscAbsScalar(p)/x->r);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ConvectionFlux"
/*
 * the flux for the convection term
 * x = (rho,rho*u,rho*v,rho*w,T)^T
 * x_t + div(Fcf + Fcg + Fch) = f
 *
   Note that here x->T is the rho scaled total energy per unit mass r*E,
   that is the total energy per unit volume
   H = E + p/r, where E is the total energy: E = r*e + 0.5*r*u*u

 * */
PetscErrorCode ConvectionFlux(User user,const PetscReal *n,const Node *x,Node *f)
{
  PetscErrorCode ierr;
  PetscScalar  u[DIM]; // unscaled u
  PetscScalar  E; // unscaled total energy
  PetscScalar  H; // unscaled stagnation
  PetscScalar  Fcf[5], Fcg[5], Fch[5];
  PetscScalar  p; //the pressure
  PetscInt     i;

  PetscFunctionBeginUser;

  /*
     Note that for the explicit method, the velocity and temperature are the so
     called density velocity and density temperature, that is, it is scalled be the density rho: x->u=r*u, x->T=T*r

     Note that here x->T is the rho scaled total energy per unit mass r*E,
     that is the total energy per unit volume, where E = r*e + 0.5*r*u*u
     but in the flux, we need the stagnation (total, enthalpy) which defined as:
     H = E + p/r,
 */

  for (i=0; i<DIM; i++) {
    u[i] = x->ru[i]/x->r;
  }
  E = x->rE/x->r;


  if(user->PressureFlux){
    ierr = Pressure_Full(user,x,&p);CHKERRQ(ierr); /* conpute the pressure */
  }else{
    p = 0.0;
  }

  H = E + p/x->r;

  Fcf[0] = x->ru[0];          Fcg[0] = x->ru[1];          Fch[0] = x->ru[2];         /* rho*u,       rho*v,       rho*w       */
  Fcf[1] = x->ru[0]*u[0] + p; Fcg[1] = x->ru[1]*u[0];     Fch[1] = x->ru[2]*u[0];    /* rho*u*u + p, rho*v*u,     rho*w*u     */
  Fcf[2] = x->ru[0]*u[1];     Fcg[2] = x->ru[1]*u[1] + p; Fch[2] = x->ru[2]*u[1];    /* rho*u*v,     rho*v*v + p, rho*w*v     */
  Fcf[3] = x->ru[0]*u[2];     Fcg[3] = x->ru[1]*u[2];     Fch[3] = x->ru[2]*u[2] + p;/* rho*u*w,     rho*v*w,     rho*w*w + p */
  Fcf[4] = x->ru[0]*H;        Fcg[4] = x->ru[1]*H;        Fch[4] = x->ru[2]*H;       /* rho*u*T,     rho*v*T,     rho*w*T     */
//printf("n=(%f, %f, %f), u=(%f, %f, %f), p = %f, H = %f, E = %f \n", n[0], n[1], n[2], x->ru[0], x->ru[1], x->ru[2], p, H, E);
  f->r     = -(n[0]*Fcf[0] + n[1]*Fcg[0] + n[2]*Fch[0]); /* for the continuty equation   */
  f->ru[0] = -(n[0]*Fcf[1] + n[1]*Fcg[1] + n[2]*Fch[1]); /* for the momentum equations x */
  f->ru[1] = -(n[0]*Fcf[2] + n[1]*Fcg[2] + n[2]*Fch[2]); /* for the momentum equations y */
  f->ru[2] = -(n[0]*Fcf[3] + n[1]*Fcg[3] + n[2]*Fch[3]); /* for the momentum equations z */
  f->rE    = -(n[0]*Fcf[4] + n[1]*Fcg[4] + n[2]*Fch[4]); /* for the energy equation      */
  //printf("f=(%f, %f, %f, %f, %f )\n", f->ru[0], f->ru[1], f->ru[2], f->rE);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DiffusionFlux"
/*
 * the flux for the diffusion term

 *
 * */
PetscErrorCode DiffusionFlux(User user, const PetscReal *cgradL, const PetscReal *cgradR, const PetscReal *fgc, const PetscReal *cgcL, const PetscReal *cgcR,
      const PetscReal *n, const Node *xL, const Node *xR, Node *f)
{
  PetscInt     i, j;
  PetscReal    graduL[DIM][DIM], graduR[DIM][DIM];
  PetscReal    gradTL[DIM], gradTR[DIM];
  PetscReal    cgcAL[DIM], cgcAR[DIM];
  PetscReal    Dfl, Dfr, tempL[DIM], tempR[DIM];
  PetscReal    D[DIM], DiffL[DIM], DiffR[DIM];
  PetscReal    gradU, gradV, gradW, gradT;

  PetscFunctionBeginUser;

  for (i=0; i<DIM; i++) {
    for (j=0; j<DIM; j++) {
      graduL[i][j] = cgradL[(i+1)*DIM + j];
      graduR[i][j] = cgradR[(i+1)*DIM + j];
    }/*gradient of the velocity*/
  }

  for (i=0; i<DIM; i++) {
    gradTL[i] = cgradL[4*DIM + i];
    gradTR[i] = cgradR[4*DIM + i];

    tempL[i] = fgc[i] - cgcL[i];
    tempR[i] = fgc[i] - cgcR[i];
  }/*gradient of the energy*/

  Dfl = DotDIM(tempL, n);
  Dfr = DotDIM(tempR, n);

//  PetscPrintf(PETSC_COMM_WORLD, "Dfl = %f, Dfr = %f\n", Dfl, Dfr);

  for (i=0; i<DIM; i++) {
    cgcAL[i] = fgc[i] - Dfl*n[i];
    cgcAR[i] = fgc[i] - Dfr*n[i];
    D[i]     = cgcAL[i] - cgcAR[i];
    DiffL[i] = cgcAL[i] - cgcL[i];
    DiffR[i] = cgcAR[i] - cgcR[i];
  }/*The auxiliary nodes (See [Ferziger and Peric (2002)])*/
  if (NormDIM(D) < 1.e-8) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"The distance of two elements is zero, check the mesh!");


  gradU = (xL->ru[0] - xR->ru[0] + DotDIM(graduL[0], DiffL) - DotDIM(graduR[0], DiffR))/NormDIM(D);
  gradV = (xL->ru[1] - xR->ru[1] + DotDIM(graduL[1], DiffL) - DotDIM(graduR[1], DiffR))/NormDIM(D);
  gradW = (xL->ru[2] - xR->ru[2] + DotDIM(graduL[2], DiffL) - DotDIM(graduR[2], DiffR))/NormDIM(D);
  gradT = (xL->rE - xR->rE + DotDIM(gradTL, DiffL) - DotDIM(gradTR, DiffR))/NormDIM(D);


  f->r = 0; /*Since the continuity equation does not have diffusion term*/

  {
    f->ru[0] = user->viscosity*gradU;
    f->ru[1] = user->viscosity*gradV;
    f->ru[2] = user->viscosity*gradW;
  }/*for the momentum equations*/

  f->rE = user->k*gradT; /*for the energy equation*/
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "RiemannSolver"
/*@C
  This function is for the Rusanov type Riemann solver.
  speed = \max\{|\mathbf{u}_L| + c_L,  |\mathbf{u}_R| + c_R \},
  where $c$ is the speed of the sound and $\mathbf{u}$ is the velocity.

  Input Parameters:
+ user  - The user defined contex
. cgradL, cgradR - The gradients at the centoid of the element on the left and right of the shared face
- fgc - The centroid coordinates of the face
+ cgcL, cgcR - The centroid coordinates of the elements on the left and right of the face
. n - The outward normal direction of the face
- xL, cR - The value of the variables at the left and right cells

  Output Parameters:
. fluxcon - The flux of the convection term on the face
. fluxdiff - The flux of the diffusion term on the face


@*/

PetscErrorCode RiemannSolver(User user, const PetscScalar *cgradL, const PetscScalar *cgradR, const PetscReal *fgc, const PetscReal *cgcL, const PetscReal *cgcR, const PetscReal *n, const PetscScalar *xL, const PetscScalar *xR, PetscScalar *fluxcon, PetscScalar *fluxdiff)
{
  PetscErrorCode  ierr;
  PetscScalar     cL,cR,speed;
  const Node      *ruL = (const Node*)xL,*ruR = (const Node*)xR;
  Node            fLcon,fRcon;
  Node            fdiff;
  PetscInt        i;
  PetscScalar     uL[DIM], uR[DIM]; // velocity

  PetscFunctionBeginUser;

  if (ruL->r < 0 || ruR->r < 0 || PetscAbsScalar(ruL->r)<1.e-5 ||  PetscAbsScalar(ruR->r)<1.e-5 ){
    ierr = PetscPrintf(PETSC_COMM_WORLD, "WORNING: density goes to negative or zero!!! rL = %f, rR = %f \n", ruL->r, ruR->r);CHKERRQ(ierr);
    //SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reconstructed density is negative or zero");
  }

  ierr = ConvectionFlux(user, n, ruL, &fLcon);CHKERRQ(ierr);
  ierr = ConvectionFlux(user, n, ruR, &fRcon);CHKERRQ(ierr);
  ierr = DiffusionFlux(user, cgradL, cgradR, fgc, cgcL, cgcR, n, ruL, ruR, &fdiff);CHKERRQ(ierr);
  ierr = SpeedOfSound_PG(user,ruL,&cL);CHKERRQ(ierr);
  ierr = SpeedOfSound_PG(user,ruR,&cR);CHKERRQ(ierr);

  for(i=0; i<DIM; i++){
    uL[i] = ruL->ru[i]/ruL->r; // unscaled velocity
    uR[i] = ruR->ru[i]/ruR->r;
  }

// the Riemann solver setup
  if(0 == strcasecmp(user->RiemannSolver, "Rusanov")){
    speed = PetscMax(cL, cR) + PetscMax(PetscAbsScalar(DotDIM(uL,n)/NormDIM(n)), PetscAbsScalar(DotDIM(uR,n)/NormDIM(n)));
    //speed = PetscMax(cL + PetscAbsScalar(DotDIM(uL->ru,n)/NormDIM(n)), cR + PetscAbsScalar(DotDIM(uR->ru,n)/NormDIM(n)));
  }else if(0 == strcasecmp(user->RiemannSolver, "David")){
    PetscScalar temp, temp1, temp2, temp3, temp4;
    temp1 = PetscAbsScalar(DotDIM(uL,n)/NormDIM(n) - cL);
    temp2 = PetscAbsScalar(DotDIM(uR,n)/NormDIM(n) - cR);
    temp3 = PetscAbsScalar(DotDIM(uL,n)/NormDIM(n) + cL);
    temp4 = PetscAbsScalar(DotDIM(uR,n)/NormDIM(n) + cR);

    temp = PetscMax(temp1, temp2);
    temp = PetscMax(temp, temp3);
    speed = PetscMax(temp, temp4);
// speed = max(temp1, temp2, temp3, temp4)
  }else if(0 == strcasecmp(user->RiemannSolver, "CFL")){
    PetscScalar dt, dx, dy, dz, dxyz;
    dt = user->dt;
    dx = PetscAbsScalar(cgcL[0] - cgcR[0]);
    dy = PetscAbsScalar(cgcL[1] - cgcR[1]);
    dz = PetscAbsScalar(cgcL[2] - cgcR[2]);

    dxyz = PetscSqrtScalar(dx*dx + dy*dy + dz*dz);
    speed = user->CFL*dxyz/dt;
  }else{
    speed = 0.0;
  }

// the convection terms
  fluxcon[0] = 0.5*(fLcon.r + fRcon.r) + 0.5*speed*(ruL->r - ruR->r);                 // the continuity equation
  fluxcon[1] = 0.5*(fLcon.ru[0] + fRcon.ru[0]) + 0.5*speed*(ruL->ru[0] - ruR->ru[0]); // the momentum equation x
  fluxcon[2] = 0.5*(fLcon.ru[1] + fRcon.ru[1]) + 0.5*speed*(ruL->ru[1] - ruR->ru[1]); // the momentum equation y
  fluxcon[3] = 0.5*(fLcon.ru[2] + fRcon.ru[2]) + 0.5*speed*(ruL->ru[2] - ruR->ru[2]); // the momentum equation z
  fluxcon[4] = 0.5*(fLcon.rE + fRcon.rE) + 0.5*speed*(ruL->rE - ruR->rE);             // the energy equation

  //printf("flux (%f, %f, %f, %f, %f)\n", fluxcon[0], fluxcon[1], fluxcon[2], fluxcon[3], fluxcon[4]);

// the diffution terms
  if (user->Euler) {
    for (i=0; i<2+DIM; i++) {
      fluxdiff[i] = 0.0;
    }
  }else{
    fluxdiff[0] = 0; /*Since the continuity equation does not have diffusion term.*/
    fluxdiff[1] = fdiff.ru[0]; // the momentum equation x
    fluxdiff[2] = fdiff.ru[1]; // the momentum equation y
    fluxdiff[3] = fdiff.ru[2]; // the momentum equation z
    fluxdiff[4] = fdiff.rE;    // the energy equation
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BoundaryInflow"
PetscErrorCode BoundaryInflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, User user)
{
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;

  if(user->benchmark_couette){
    ierr = ExactSolution(time, c, xG, user);CHKERRQ(ierr);
  }else{
    xG[0] = 1.0; /*Density*/
    xG[1] = user->inflow_u; /*Velocity u (the x-direction)*/
    xG[2] = user->inflow_v; /*Velocity v (the y-direction)*/
    xG[3] = user->inflow_w; /*Velocity w (the z-direction)*/
    xG[4] = 1.0; /*Energy*/
  }
  //printf("inlet: (%f, %f, %f)\n", c[0], c[1], c[2]);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BoundaryOutflow"
PetscErrorCode BoundaryOutflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, User user)
{
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;

  if(user->benchmark_couette){
    ierr = ExactSolution(time, c, xG, user);CHKERRQ(ierr);
  }else{
    xG[0] = 1.0; /*Density*/
    xG[1] = 1.0; /*Velocity u (the x-direction)*/
    xG[2] = 1.0; /*Velocity v (the y-direction)*/
    xG[3] = 1.0; /*Velocity w (the z-direction)*/
    xG[4] = 1.0; /*Energy*/
  }
  //printf("outlet: (%f, %f, %f)\n", c[0], c[1], c[2]);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BoundaryWallflow"
PetscErrorCode BoundaryWallflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, User user)
{
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;

  if(user->benchmark_couette){
    ierr = ExactSolution(time, c, xG, user);CHKERRQ(ierr);
  }else{
    PetscScalar xn[DIM],xt[DIM];
    NormalSplitDIM(n,xI+1,xn,xt);
    xG[0] = 1.0; /*Density*/
    xG[1] = 1.0; /*Velocity u (the x-direction)*/
    xG[2] = 1.0; /*Velocity v (the y-direction)*/
    xG[3] = 1.0; /*Velocity w (the z-direction)*/
    xG[4] = 1.0; /*Energy*/
  }
  //printf("wall: (%f, %f, %f)\n", c[0], c[1], c[2]);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InitialCondition"
PetscErrorCode InitialCondition(PetscReal time, const PetscReal *x, PetscScalar *u, User user)
{
  PetscInt i;

  PetscFunctionBeginUser;
  if (time != 0.0) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No solution known for time %g",time);
  if(user->benchmark_couette){

    PetscScalar U, H, T;
    PetscScalar y;

    y = x[1];

    U = 0.3; H = 10;
    T = user->T0 + (y/H)*(user->T1 - user->T0) + (user->viscosity*U*U)/(2*user->k)*(y/H)*(1.0 - y/H);

    u[0] = 1.0/T; // Density
    u[1] = u[0]*y*U/H + 0.001; // Velocity rho*u (the x-direction)
    u[2] = 0.0; // Velocity v (the y-direction)
    u[3] = 0.0; // Velocity w (the z-direction)
    u[4] = 1.0/(user->adiabatic*(user->adiabatic-1)) + 0.5*(u[1]*u[1]+u[2]*u[2]+u[3]*u[3])/u[0]; //the density total energy rho*E = rho*e + 0.5*rho*|u|^2 = rho*(p/(rho*(gamma-1))) + 0.5*rho*|u|^2

    u[1] = 0;

    //PetscErrorCode  ierr;
    //ierr = ExactSolution(time, x, u, user);CHKERRQ(ierr);
  }else{
    u[0]     = 1.0; /* Density */
    u[DIM+1] = 1.0; /* Energy */
    for (i=1; i<DIM+1; i++) u[i] = 1.0; /* Momentum */
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMPlexGetIndex"
/*@C
  DMPlexGetIndex - Get the index of 'point'. It returns the global indices.

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or NULL to use the default section
. globalSection - The section describing the layout in v, or NULL to use the default global section
. point - The sieve point in the DM
. NumOfIndices - The number of the indices for this point
. indices - The indices of this point

  Level: intermediate

.seealso DMPlexVecSetClosure()
@*/
PetscErrorCode DMPlexGetIndex(DM dm, PetscSection section, PetscSection globalSection, PetscInt point, PetscInt *NumOfIndices, PetscInt index[])
{
  PetscSection    clSection;
  IS              clPoints;
  PetscInt       *points = NULL;
  const PetscInt *clp;
  PetscInt       *indices;
  PetscInt        offsets[32];
  PetscInt        numFields, numIndices, numPoints, dof, off, globalOff, pStart, pEnd, p, q, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  if (!globalSection) {ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(globalSection, PETSC_SECTION_CLASSID, 3);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscSectionGetClosureIndex(section, (PetscObject) dm, &clSection, &clPoints);CHKERRQ(ierr);
  if (!clPoints) {
    ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    /* Compress out points not in the section */
    ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = 0, q = 0; p < numPoints*2; p += 2) {
      if ((points[p] >= pStart) && (points[p] < pEnd)) {
        points[q*2]   = points[p];
        points[q*2+1] = points[p+1];
        ++q;
      }
    }
    numPoints = q;
  } else {
    PetscInt dof, off;

    ierr = PetscSectionGetDof(clSection, point, &dof);CHKERRQ(ierr);
    numPoints = dof/2;
    ierr = PetscSectionGetOffset(clSection, point, &off);CHKERRQ(ierr);
    ierr = ISGetIndices(clPoints, &clp);CHKERRQ(ierr);
    points = (PetscInt *) &clp[off];
  }
  for (p = 0, numIndices = 0; p < numPoints*2; p += 2) {
    PetscInt fdof;

    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      ierr          = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
      offsets[f+1] += fdof;
    }
    numIndices += dof;
  }
  for (f = 1; f < numFields; ++f) offsets[f+1] += offsets[f];

  if (numFields && offsets[numFields] != numIndices) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Invalid size for closure %d should be %d", offsets[numFields], numIndices);
  ierr = DMGetWorkArray(dm, numIndices, PETSC_INT, &indices);CHKERRQ(ierr);
  if (numFields) {
    for (p = 0; p < numPoints*2; p += 2) {
      PetscInt o = points[p+1];
      ierr = PetscSectionGetOffset(globalSection, points[p], &globalOff);CHKERRQ(ierr);
      indicesPointFields_private(section, points[p], globalOff < 0 ? -(globalOff+1) : globalOff, offsets, PETSC_FALSE, o, indices);
    }
  } else {
    for (p = 0, off = 0; p < numPoints*2; p += 2) {
      PetscInt o = points[p+1];
      ierr = PetscSectionGetOffset(globalSection, points[p], &globalOff);CHKERRQ(ierr);
      indicesPoint_private(section, points[p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, o, indices);
    }
  }

  if (!clPoints) {
    ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
  } else {
    ierr = ISRestoreIndices(clPoints, &clp);CHKERRQ(ierr);
  }

  {
    /*Out put the number of indices and the value of indices on this point*/
    PetscInt j;
    *NumOfIndices = numIndices;
    for(j = 0; j < numIndices; j++){
      index[j] = indices[j];
    }
  }
  ierr = DMRestoreWorkArray(dm, numIndices, PETSC_INT, &indices);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMonitorFunctionError"
/*@C
   TSMonitorFunctionError - Monitors progress of the TS solvers by the solution norm

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time

@*/
PetscErrorCode  TSMonitorFunctionError(TS ts,PetscInt step,PetscReal ptime,Vec u,void *ctx)
{
  User              user = (User) ctx;

  PetscErrorCode    ierr;
  PetscReal         norm, funcnorm;
  //PetscLogDouble    space =0;
  PetscInt          size;
  Vec               func;
  PetscInt          nplot = 0;
  char              fileName[2048];

  PetscFunctionBegin;
  if (step%10==0) {
      ierr = VecDuplicate(u, &func);CHKERRQ(ierr);
      ierr = TSComputeRHSFunction(ts, ptime, u, func);CHKERRQ(ierr);
      ierr = VecNorm(func,NORM_2,&funcnorm);CHKERRQ(ierr);
      //ierr = VecView(func, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = VecDestroy(&func);CHKERRQ(ierr);
      ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
      ierr = VecGetSize(u, &size);CHKERRQ(ierr);
      norm      = norm/size;
      funcnorm  = funcnorm/size;
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Step %D at time %g with solution norm = %g right hand side function norm = %g\n",step, ptime,norm, funcnorm);CHKERRQ(ierr);
      //ierr =  PetscMallocGetCurrentUsage(&space);CHKERRQ(ierr);
      //ierr =  PetscPrintf(PETSC_COMM_WORLD,"Current space PetscMalloc()ed %g M\n", space/(1024*1024));CHKERRQ(ierr);
  }
  user->current_time = ptime;
  // output the solution
  if (user->output_solution && (step%user->steps_output==0)){
    PetscViewer    viewer;
    Vec            solution_unscaled; // Note the the u is scaled by the density, so this is for the unscaled solution

    nplot = step/user->steps_output;
    // update file name for the current time step
    ierr = VecDuplicate(u, &solution_unscaled);CHKERRQ(ierr);
    ierr = ReformatSolution(u, solution_unscaled, user);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fileName, sizeof(fileName),"%s_%d.vtk",user->solutionfile, nplot);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Outputing solution %s (current time %f)\n", fileName, ptime);CHKERRQ(ierr);
    ierr = OutputVTK(user->dm, fileName, &viewer);CHKERRQ(ierr);
    ierr = VecView(solution_unscaled, viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&solution_unscaled);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ExactSolution"
PetscErrorCode ExactSolution(PetscReal time, const PetscReal *c, PetscScalar *xc, User user)
{
  PetscScalar U, H, T, u;
  PetscScalar y;

  PetscFunctionBeginUser;

  y = c[1];
  U = 0.3; H = 10;

  u = y*U/H;
  T = user->T0 + (y/H)*(user->T1 - user->T0) + (user->viscosity*U*U)/(2*user->k)*(y/H)*(1.0 - y/H);
  // T = 1;

  xc[0] = 1.0/T; /*Density*/
  xc[1] = xc[0]*u; /*Velocity rho*u (the x-direction)*/
  xc[2] = 0.0; /*Velocity v (the y-direction)*/
  xc[3] = 0.0; /*Velocity w (the z-direction)*/
  xc[4] = 1.0/(user->adiabatic*(user->adiabatic-1)) + 0.5*(xc[1]*xc[1]+xc[2]*xc[2]+xc[3]*xc[3])/xc[0]; /*the density total energy rho*E = rho*e + 0.5*rho*|u|^2 = rho*(p/(rho*(gamma-1))) + 0.5*rho*|u|^2*/

  PetscFunctionReturn(0);
}

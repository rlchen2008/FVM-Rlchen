#include <petscts.h>
#include <petscfv.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <petscblaslapack.h>
#include "AeroSim.h"


#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
/**
   The second heart of the nonlinear solver.  Assembles Jacobian matrix.
   If nonlinear solves are not converging, this is the place to start looking.

   @param snes nonlinear solver context
   @param g point at which Jacobian is to be evaluated
   @param jac matrix pointer to put values in
   @param B preconditioner pointer to put values in (usually == jac)
*/
PetscErrorCode FormJacobian(SNES snes, Vec g, Mat jac, Mat B, void *ctx)
{
  User user = (User) ctx;

  //Algebra algebra = user->algebra;

  PetscErrorCode ierr;
  PetscMPIInt rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  //ierr = MatCreateSNESMF(snes,&jac);CHKERRQ(ierr);

  if (user->fd_jacobian) {  /* compute the Jacobian using FD */
//    PetscPrintf(PETSC_COMM_WORLD,"Form Jacobian\n");
    ierr = SNESComputeJacobianDefault(snes, g, jac, jac, (void*) ctx);CHKERRQ(ierr);
    ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1.e6);CHKERRQ(ierr);
//    MatView(jac,PETSC_VIEWER_STDOUT_WORLD);
  }else if(user->fd_jacobian_color){ /* compute the Jacobian using FD coloring */
    /* using the FD coloring to find the jacobian of the stabilizd term
       and form the analytic jacobian of the linear and nonlinear term */
    ierr = SNESComputeJacobianDefaultColor(snes, g, jac, jac, 0);CHKERRQ(ierr);
  }else {
    /* form the analytic Jacobian for all the terms */
    ierr = SetupJacobian(user->dm, g, jac, B, user);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupJacobian"
PetscErrorCode SetupJacobian(DM dm, Vec X, Mat jac, Mat B, void *ctx)
{
  User              user = (User) ctx;
  Physics           phys = user->model->physics;
  PetscSection      section, globalSection;

  PetscInt          cStart, cEnd, c;
//  PetscInt          numCells;
  PetscInt          dof = phys->dof;
  PetscErrorCode    ierr;
  Vec               inLocal;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "dof = %d\n", dof);
  ierr = DMGetLocalVector(user->dm, &inLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->dm, X, INSERT_VALUES, inLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dm, X, INSERT_VALUES, inLocal);CHKERRQ(ierr);

  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  cEnd = user->cEndInterior;
  //numCells = cEnd - cStart;

  {
    PetscInt         NumOfIndices;
    PetscInt         indices[dof];
    PetscScalar      *values;

    for (c = cStart; c < cEnd; ++c) {
      ierr = DMPlexGetIndex(dm, section, globalSection, c, &NumOfIndices, indices);CHKERRQ(ierr);
      ierr = PetscMalloc1(NumOfIndices*NumOfIndices, &values);CHKERRQ(ierr);
      ierr = PetscMemzero(values, NumOfIndices*NumOfIndices* sizeof(PetscScalar));CHKERRQ(ierr);

      if (user->second_order){
        ierr = ComputeJacobian_LS(dm, inLocal, c, values, user);CHKERRQ(ierr);
      }else{
        ierr = ComputeJacobian_Upwind(dm, inLocal, c, values, user);CHKERRQ(ierr);
      }

      ierr = MatSetValues(jac, NumOfIndices, indices, NumOfIndices, indices, values, INSERT_VALUES);
      ierr = PetscFree(values);CHKERRQ(ierr);
    }
  }


  ierr = DMLocalToGlobalBegin(user->dm, inLocal, INSERT_VALUES, X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->dm, inLocal, INSERT_VALUES, X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &inLocal);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  {
    PetscViewer viewer;
    char filename[256];
    sprintf(filename,"matJac.m");
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,
                              &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\n% -----------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "%  Matrix Jacobian: \n% -------------------------\n");CHKERRQ(ierr);
    ierr = MatView(jac, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeJacobian_Upwind"
PetscErrorCode ComputeJacobian_Upwind(DM dm, Vec locX, PetscInt cell, PetscScalar CellValues[], void *ctx)
{
  User              user = (User) ctx;
  Physics           phys = user->model->physics;
  PetscInt          dof = phys->dof;
  const PetscScalar *facegeom, *cellgeom,*x;
  PetscErrorCode    ierr;
  DM                dmFace, dmCell;

  DM                dmGrad = user->dmGrad;
  PetscInt          fStart, fEnd, face, cStart;
  Vec               locGrad, locGradLimiter, Grad;
  /*here the localGradLimiter refers to the gradient that has been multiplied by the limiter function.
   The locGradLimiter is used to construct the uL and uR, and the locGrad is used to caculate the diffusion term*/
  Vec               TempVec; /*a temperal vec for the vector restore*/

  PetscFunctionBeginUser;

  ierr = VecGetDM(user->facegeom,&dmFace);CHKERRQ(ierr);
  ierr = VecGetDM(user->cellgeom,&dmCell);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);
  ierr = VecDuplicate(Grad, &TempVec);CHKERRQ(ierr);
  ierr = VecCopy(Grad, TempVec);CHKERRQ(ierr);
  /*Backup the original vector and use it to restore the value of dmGrad,
    because I do not want to change the values of the cell gradient*/

  ierr = VecGetArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  {
    PetscScalar *grad;
    ierr = VecGetArray(Grad,&grad);CHKERRQ(ierr);

    /* Limit interior gradients. Using cell-based loop because it generalizes better to vector limiters. */

      const PetscInt    *faces;
      PetscInt          numFaces,f;
      PetscReal         *cellPhi; /* Scalar limiter applied to each component separately */
      const PetscScalar *cx;
      const CellGeom    *cg;
      PetscScalar       *cgrad;
      PetscInt          i;

      ierr = PetscMalloc(phys->dof*sizeof(PetscScalar),&cellPhi);CHKERRQ(ierr);

      ierr = DMPlexGetConeSize(dm,cell,&numFaces);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm,cell,&faces);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dm,cell,x,&cx);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell,cell,cellgeom,&cg);CHKERRQ(ierr);
      ierr = DMPlexPointGlobalRef(dmGrad,cell,grad,&cgrad);CHKERRQ(ierr);

      /* Limiter will be minimum value over all neighbors */
      for (i=0; i<dof; i++) {
        cellPhi[i] = PETSC_MAX_REAL;
      }
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
        ncell = cell == fcells[0] ? fcells[1] : fcells[0];  /*The expression (x ? y : z) has the value of y if x is nonzero, z otherwise */
        ierr  = DMPlexPointLocalRead(dm,ncell,x,&ncx);CHKERRQ(ierr);
        ierr  = DMPlexPointLocalRead(dmCell,ncell,cellgeom,&ncg);CHKERRQ(ierr);
        Waxpy2(-1, cg->centroid, ncg->centroid, v);
        for (i=0; i<dof; i++) {
          /* We use the symmetric slope limited form of Berger, Aftosmis, and Murman 2005 */
          PetscScalar phi,flim = 0.5 * (ncx[i] - cx[i]) / Dot2(&cgrad[i*DIM],v);
          phi        = (*user->LimitGrad)(flim);
          cellPhi[i] = PetscMin(cellPhi[i],phi);
        }
      }
      /* Apply limiter to gradient */
      for (i=0; i<dof; i++) Scale2(cellPhi[i],&cgrad[i*DIM],&cgrad[i*DIM]);

      ierr = PetscFree(cellPhi);CHKERRQ(ierr);

    ierr = VecRestoreArray(Grad,&grad);CHKERRQ(ierr);
  }
  ierr = DMGetLocalVector(dmGrad,&locGradLimiter);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmGrad,Grad,INSERT_VALUES,locGradLimiter);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmGrad,Grad,INSERT_VALUES,locGradLimiter);CHKERRQ(ierr);

  ierr = VecCopy(TempVec, Grad);CHKERRQ(ierr);/*Restore the vector*/

  ierr = DMGetLocalVector(dmGrad,&locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmGrad,Grad,INSERT_VALUES,locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmGrad,Grad,INSERT_VALUES,locGrad);CHKERRQ(ierr);

  ierr = DMRestoreGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);
  ierr = VecDestroy(&TempVec);CHKERRQ(ierr);

  {
    const PetscScalar *grad, *gradlimiter;
    ierr = VecGetArrayRead(locGrad,&grad);CHKERRQ(ierr);
    ierr = VecGetArrayRead(locGradLimiter,&gradlimiter);CHKERRQ(ierr);
    for (face=fStart; face<fEnd; face++) {
      const PetscInt    *cells;
      PetscInt          ghost,i,j;
      PetscScalar       *fluxcon, *fluxdiff, *fx[2];
      const FaceGeom    *fg;
      const CellGeom    *cg[2];
      const PetscScalar *cx[2],*cgrad[2], *cgradlimiter[2];
      PetscScalar       *uL, *uR;
      PetscReal         FaceArea;

      ierr = PetscMalloc(phys->dof * phys->dof * sizeof(PetscScalar), &fluxcon);CHKERRQ(ierr); /*For the convection terms*/
      ierr = PetscMalloc(phys->dof * phys->dof * sizeof(PetscScalar), &fluxdiff);CHKERRQ(ierr); /*For the diffusion terms*/
      ierr = PetscMalloc(phys->dof * sizeof(PetscScalar), &uL);CHKERRQ(ierr);
      ierr = PetscMalloc(phys->dof * sizeof(PetscScalar), &uR);CHKERRQ(ierr);

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
        Waxpy2(-1,cg[i]->centroid,fg->centroid,dx);
        for (j=0; j<dof; j++) {
          fx[i][j] = cx[i][j] + Dot2(cgradlimiter[i],dx);
        }
        /*fx[0] and fx[1] are the value of the variables on the left and right
          side of the face, respectively, that is u_L and u_R.*/
      }

      ierr = RiemannSolver_Rusanov_Jacobian(user, cgrad[0], cgrad[1], fg->centroid, cg[0]->centroid, cg[1]->centroid, fg->normal,
                  fx[0], fx[1], fluxcon, fluxdiff);CHKERRQ(ierr);

      ierr = DMPlexComputeCellGeometryFVM(dm, face, &FaceArea, NULL, NULL);CHKERRQ(ierr);
        /*Compute the face area*/

      for (i=0; i<phys->dof; i++) {
        for (j=0; j<phys->dof; j++) {
          if(cells[0]<user->cEndInterior) CellValues[cells[0]*dof*dof + i*dof + j] -= cells[0]*1.0;
          if(cells[1]<user->cEndInterior) CellValues[cells[1]*dof*dof + i*dof + j] += cells[1]*1.2;
        }
      }
//      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
      ierr = PetscFree(fluxcon);CHKERRQ(ierr);
      ierr = PetscFree(fluxdiff);CHKERRQ(ierr);
      ierr = PetscFree(uL);CHKERRQ(ierr);
      ierr = PetscFree(uR);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(locGrad,&grad);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locGradLimiter,&gradlimiter);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmGrad,&locGradLimiter);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmGrad,&locGrad);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeJacobian_LS"
PetscErrorCode ComputeJacobian_LS(DM dm, Vec locX, PetscInt cell, PetscScalar CellValues[], void *ctx)
{
  User              user = (User) ctx;
  Physics           phys = user->model->physics;
  PetscInt          dof = phys->dof;
  const PetscScalar *facegeom, *cellgeom,*x;
  PetscErrorCode    ierr;
  DM                dmFace, dmCell;

  DM                dmGrad = user->dmGrad;
  PetscInt          fStart, fEnd, face, cStart;
  Vec               locGrad, locGradLimiter, Grad;
  /*here the localGradLimiter refers to the gradient that has been multiplied by the limiter function.
   The locGradLimiter is used to construct the uL and uR, and the locGrad is used to caculate the diffusion term*/
  Vec               TempVec; /*a temperal vec for the vector restore*/

  PetscFunctionBeginUser;

  ierr = VecGetDM(user->facegeom,&dmFace);CHKERRQ(ierr);
  ierr = VecGetDM(user->cellgeom,&dmCell);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);
  ierr = VecDuplicate(Grad, &TempVec);CHKERRQ(ierr);
  ierr = VecCopy(Grad, TempVec);CHKERRQ(ierr);
  /*Backup the original vector and use it to restore the value of dmGrad,
    because I do not want to change the values of the cell gradient*/

  ierr = VecGetArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  {
    PetscScalar *grad;
    ierr = VecGetArray(Grad,&grad);CHKERRQ(ierr);

    /* Limit interior gradients. Using cell-based loop because it generalizes better to vector limiters. */

      const PetscInt    *faces;
      PetscInt          numFaces,f;
      PetscReal         *cellPhi; /* Scalar limiter applied to each component separately */
      const PetscScalar *cx;
      const CellGeom    *cg;
      PetscScalar       *cgrad;
      PetscInt          i;

      ierr = PetscMalloc(phys->dof*sizeof(PetscScalar),&cellPhi);CHKERRQ(ierr);

      ierr = DMPlexGetConeSize(dm,cell,&numFaces);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm,cell,&faces);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dm,cell,x,&cx);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell,cell,cellgeom,&cg);CHKERRQ(ierr);
      ierr = DMPlexPointGlobalRef(dmGrad,cell,grad,&cgrad);CHKERRQ(ierr);

      /* Limiter will be minimum value over all neighbors */
      for (i=0; i<dof; i++) {
        cellPhi[i] = PETSC_MAX_REAL;
      }
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
        ncell = cell == fcells[0] ? fcells[1] : fcells[0];  /*The expression (x ? y : z) has the value of y if x is nonzero, z otherwise */
        ierr  = DMPlexPointLocalRead(dm,ncell,x,&ncx);CHKERRQ(ierr);
        ierr  = DMPlexPointLocalRead(dmCell,ncell,cellgeom,&ncg);CHKERRQ(ierr);
        Waxpy2(-1, cg->centroid, ncg->centroid, v);
        for (i=0; i<dof; i++) {
          /* We use the symmetric slope limited form of Berger, Aftosmis, and Murman 2005 */
          PetscScalar phi,flim = 0.5 * (ncx[i] - cx[i]) / Dot2(&cgrad[i*DIM],v);
          phi        = (*user->LimitGrad)(flim);
          cellPhi[i] = PetscMin(cellPhi[i],phi);
        }
      }
      /* Apply limiter to gradient */
      for (i=0; i<dof; i++) Scale2(cellPhi[i],&cgrad[i*DIM],&cgrad[i*DIM]);

      ierr = PetscFree(cellPhi);CHKERRQ(ierr);

    ierr = VecRestoreArray(Grad,&grad);CHKERRQ(ierr);
  }
  ierr = DMGetLocalVector(dmGrad,&locGradLimiter);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmGrad,Grad,INSERT_VALUES,locGradLimiter);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmGrad,Grad,INSERT_VALUES,locGradLimiter);CHKERRQ(ierr);

  ierr = VecCopy(TempVec, Grad);CHKERRQ(ierr);/*Restore the vector*/

  ierr = DMGetLocalVector(dmGrad,&locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmGrad,Grad,INSERT_VALUES,locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmGrad,Grad,INSERT_VALUES,locGrad);CHKERRQ(ierr);

  ierr = DMRestoreGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);
  ierr = VecDestroy(&TempVec);CHKERRQ(ierr);

  {
    const PetscScalar *grad, *gradlimiter;
    ierr = VecGetArrayRead(locGrad,&grad);CHKERRQ(ierr);
    ierr = VecGetArrayRead(locGradLimiter,&gradlimiter);CHKERRQ(ierr);
    for (face=fStart; face<fEnd; face++) {
      const PetscInt    *cells;
      PetscInt          ghost,i,j;
      PetscScalar       *fluxcon, *fluxdiff, *fx[2];
      const FaceGeom    *fg;
      const CellGeom    *cg[2];
      const PetscScalar *cx[2],*cgrad[2], *cgradlimiter[2];
      PetscScalar       *uL, *uR;
      PetscReal         FaceArea;

      ierr = PetscMalloc(phys->dof * phys->dof * sizeof(PetscScalar), &fluxcon);CHKERRQ(ierr); /*For the convection terms*/
      ierr = PetscMalloc(phys->dof * phys->dof * sizeof(PetscScalar), &fluxdiff);CHKERRQ(ierr); /*For the diffusion terms*/
      ierr = PetscMalloc(phys->dof * sizeof(PetscScalar), &uL);CHKERRQ(ierr);
      ierr = PetscMalloc(phys->dof * sizeof(PetscScalar), &uR);CHKERRQ(ierr);

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
        Waxpy2(-1,cg[i]->centroid,fg->centroid,dx);
        for (j=0; j<dof; j++) {
          fx[i][j] = cx[i][j] + Dot2(cgradlimiter[i],dx);
        }
        /*fx[0] and fx[1] are the value of the variables on the left and right
          side of the face, respectively, that is u_L and u_R.*/
      }

      ierr = RiemannSolver_Rusanov_Jacobian(user, cgrad[0], cgrad[1], fg->centroid, cg[0]->centroid, cg[1]->centroid, fg->normal,
                  fx[0], fx[1], fluxcon, fluxdiff);CHKERRQ(ierr);

      ierr = DMPlexComputeCellGeometryFVM(dm, face, &FaceArea, NULL, NULL);CHKERRQ(ierr);
        /*Compute the face area*/

      for (i=0; i<phys->dof; i++) {
        for (j=0; j<phys->dof; j++) {
          if(cells[0]<user->cEndInterior) CellValues[cells[0]*dof*dof + i*dof + j] -= cells[0]*1.0;
          if(cells[1]<user->cEndInterior) CellValues[cells[1]*dof*dof + i*dof + j] += cells[1]*1.2;
        }
      }
//      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
      ierr = PetscFree(fluxcon);CHKERRQ(ierr);
      ierr = PetscFree(fluxdiff);CHKERRQ(ierr);
      ierr = PetscFree(uL);CHKERRQ(ierr);
      ierr = PetscFree(uR);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(locGrad,&grad);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locGradLimiter,&gradlimiter);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmGrad,&locGradLimiter);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmGrad,&locGrad);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



/* PetscReal* => Node* conversion */
#undef __FUNCT__
#define __FUNCT__ "RiemannSolver_Rusanov_Jacobian"
/*
This function is for the Rusanov type Riemann solver.
speed = \max\{|\mathbf{u}_L| + c_L,  |\mathbf{u}_R| + c_R \},
where $c$ is the speed of the sound and $\mathbf{u}$ is the velocity.
*/
PetscErrorCode RiemannSolver_Rusanov_Jacobian(User user, const PetscScalar *cgradL, const PetscScalar *cgradR,
                                             const PetscReal *fgc, const PetscReal *cgcL, const
                                             PetscReal *cgcR, const PetscReal *n, const PetscScalar *xL, const PetscScalar *xR,
                                             PetscScalar *fluxcon, PetscScalar *fluxdiff)
{
  PetscErrorCode  ierr;
  PetscScalar     cL,cR,speed;
  const Node      *uL = (const Node*)xL,*uR = (const Node*)xR;
  Node            fLcon,fRcon;
  Node            fLdiff,fRdiff;
  PetscInt        i, j;
  Physics         phys = user->model->physics;
  PetscInt        dof = phys->dof;

  PetscFunctionBeginUser;
//  if (uL->r < 0 || uR->r < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reconstructed density is negative");
  ierr = ConvectionFlux(user, n, uL, &fLcon);CHKERRQ(ierr);
  ierr = ConvectionFlux(user, n, uR, &fRcon);CHKERRQ(ierr);
  ierr = DiffusionFlux(user, cgradL, cgradR, fgc, cgcL, cgcR, n, uL, uR, &fLdiff);CHKERRQ(ierr);
  ierr = SpeedOfSound_PG(user,uL,&cL);CHKERRQ(ierr);
  ierr = SpeedOfSound_PG(user,uR,&cR);CHKERRQ(ierr);
  speed = PetscMax(cL,cR) + PetscMax(PetscAbsScalar(DotDIM(uL->ru,n)/NormDIM(n)),PetscAbsScalar(DotDIM(uR->ru,n)/NormDIM(n)));
  speed =10;
//  PetscPrintf(PETSC_COMM_WORLD, "normal = %f\n", NormDIM(n));
  for (i=0; i<dof; i++) {
    for (j=0; j<dof; j++) {
      fluxcon[dof*i + j] = 0.5*(fLcon.vals[i]+fRcon.vals[i])+0.5*speed*(xL[i]-xR[i]);
    }
  }
  for (i=0; i<dof; i++) {
    for (j=0; j<dof; j++) {
      fluxdiff[dof*i + j] = 0.5*(fLdiff.vals[i]+fRdiff.vals[i])+0.5*speed*(xL[i]-xR[i]);
    }
  }
  fluxdiff[0] = 0; /*Since the continuity equation does not have diffusion term.*/

//  for (i=0; i<phys->dof; i++) {
//    for (j=0; j<phys->dof; j++) {
//      if(cells[0]<user->cEndInterior) elemMat[cells[0]*dof*dof + i*dof + j] -= cells[0]*1.0;
//      if(cells[1]<user->cEndInterior) elemMat[cells[1]*dof*dof + i*dof + j] += cells[1]*1.2;
//    }
//  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GradientGradientJacobian"
/**
Compute the gadient of the cell center gradient obtained by the least-square method
*/
PetscErrorCode GradientGradientJacobian(DM dm, Vec locX, PetscScalar elemMat[], void *ctx)
{
  User              user = (User) ctx;
  Physics           phys = user->model->physics;
  PetscInt          dof = phys->dof;
  const PetscScalar *facegeom, *cellgeom,*x;
  PetscErrorCode    ierr;
  DM                dmFace, dmCell;

  DM                dmGrad = user->dmGrad;
  PetscInt          fStart, fEnd, face, cStart;
  Vec               Grad;
  /*here the localGradLimiter refers to the gradient that has been multiplied by the limiter function.
   The locGradLimiter is used to construct the uL and uR, and the locGrad is used to caculate the diffusion term*/
  Vec               TempVec; /*a temperal vec for the vector restore*/

  PetscFunctionBeginUser;

  ierr = VecGetDM(user->facegeom,&dmFace);CHKERRQ(ierr);
  ierr = VecGetDM(user->cellgeom,&dmCell);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);
  ierr = VecZeroEntries(Grad);CHKERRQ(ierr);
  ierr = VecDuplicate(Grad, &TempVec);CHKERRQ(ierr);
  ierr = VecCopy(Grad, TempVec);CHKERRQ(ierr);

  ierr = VecGetArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  {
    PetscScalar *grad;
    ierr = VecGetArray(TempVec,&grad);CHKERRQ(ierr);
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
      for (i=0; i<phys->dof; i++) {
        for (j=0; j<phys->dof; j++) {
          if(cells[0]<user->cEndInterior) elemMat[cells[0]*dof*dof + i*dof + j] -= cells[0]*1.0;
          if(cells[1]<user->cEndInterior) elemMat[cells[1]*dof*dof + i*dof + j] += cells[1]*1.2;
        }
      }
    }
    ierr = VecRestoreArray(TempVec,&grad);CHKERRQ(ierr);
  }
  ierr = DMRestoreGlobalVector(dmGrad,&Grad);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(locX,&x);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ApplyBC"
PetscErrorCode ApplyBC(DM dm, PetscReal time, Vec locX, User user)
{
  const char        *name = "Face Sets"; /*Set up in the function DMPlexCreateExodus. is the side set*/
  DM                dmFace;
  IS                idIS;
  const PetscInt    *ids;
  PetscScalar       *x;
  const PetscScalar *facegeom;
  PetscInt          numFS, fs;
  PetscErrorCode    ierr;
  PetscMPIInt       rank;

  PetscFunctionBeginUser;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = VecGetDM(user->facegeom,&dmFace);CHKERRQ(ierr);
  ierr = DMPlexGetLabelIdIS(dm, name, &idIS);CHKERRQ(ierr);
 // ISView(idIS, PETSC_VIEWER_STDOUT_SELF);
  if (!idIS) PetscFunctionReturn(0);
  ierr = ISGetLocalSize(idIS, &numFS);CHKERRQ(ierr);
  ierr = ISGetIndices(idIS, &ids);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->facegeom, &facegeom);CHKERRQ(ierr);
  ierr = VecGetArray(locX, &x);CHKERRQ(ierr);

  for (fs = 0; fs < numFS; ++fs) {
    IS               faceIS;
    const PetscInt   *faces;
    PetscInt         numFaces, f;

    ierr = DMPlexGetStratumIS(dm, name, ids[fs], &faceIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
//      PetscPrintf(PETSC_COMM_SELF, "rank[%d]: ids[%d] = %d, faceIS[%d] = %d, numFaces = %d\n", rank, fs, ids[fs], f, faces[f], numFaces);
      const PetscInt    face = faces[f], *cells;
      const PetscScalar *xI; /*Inner point*/
      PetscScalar       *xG; /*Ghost point*/
      const FaceGeom    *fg;

      ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dm, cells[0], x, &xI);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRef(dm, cells[1], x, &xG);CHKERRQ(ierr);
      if (ids[fs]==1){
        //PetscPrintf(PETSC_COMM_SELF, "Set Inlfow Boundary Condition! \n");
        ierr = BoundaryInflow(time, fg->centroid, fg->normal, xI, xG, user);CHKERRQ(ierr);
//        DM                dmCell;
//        const PetscScalar *cellgeom;
//        const CellGeom    *cgL, *cgR;
//        ierr = VecGetDM(user->cellgeom,&dmCell);CHKERRQ(ierr);
//        ierr = VecGetArrayRead(user->cellgeom, &cellgeom);CHKERRQ(ierr);
//        ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL);CHKERRQ(ierr);
//        ierr = DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR);CHKERRQ(ierr);
//        ierr = PetscPrintf(PETSC_COMM_WORLD,"cells[0] = (%f, %f, %f), cells[1] = (%f, %f, %f)\n",cgL->centroid[0], cgL->centroid[1], cgL->centroid[2],cgR->centroid[0], cgR->centroid[1], cgR->centroid[2]);CHKERRQ(ierr);
      }else if (ids[fs]==2){
        //PetscPrintf(PETSC_COMM_SELF, "Set Outlfow Boundary Condition! \n");
        ierr = BoundaryOutflow(time, fg->centroid, fg->normal, xI, xG, user);CHKERRQ(ierr);
      }else if (ids[fs]==3){
        //PetscPrintf(PETSC_COMM_SELF, "Set Wall Boundary Condition! \n");
        ierr = BoundaryWallflow(time, fg->centroid, fg->normal, xI, xG, user);CHKERRQ(ierr);
      }else {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Wrong type of boundary condition setup!!! \n The set up of the boundary should be: 1 for the inflow, 2 for the outflow, and 3 for the wallflow");
      }
    }
//    PetscPrintf(PETSC_COMM_SELF, " \n");
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(locX, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = ISRestoreIndices(idIS, &ids);CHKERRQ(ierr);
  ierr = ISDestroy(&idIS);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

//
//#undef __FUNCT__
//#define __FUNCT__ "ComputeExactSolution"
//PetscErrorCode ComputeExactSolution(DM dm, Vec locX, User user)
//{
//  PetscErrorCode    ierr;
//  const PetscScalar *facegeom, *cellgeom, *x;
//  PetscScalar       *f;
//  PetscInt          fStart, fEnd, face;
//  Physics           phys = user->model->physics;
//  DM                dmFace, dmCell;
//
//
//  PetscFunctionBeginUser;
//
//  ierr = VecGetDM(user->facegeom,&dmFace);CHKERRQ(ierr);
//  ierr = VecGetDM(user->cellgeom,&dmCell);CHKERRQ(ierr);
//
//  ierr = VecGetArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
//  ierr = VecGetArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
//  ierr = VecGetArrayRead(locX,&x);CHKERRQ(ierr);
//
//  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
//
//  {
//    const PetscInt    *cells;
//    PetscInt          i,ghost;
//    PetscScalar       *fL,*fR;
//    const FaceGeom    *fg;
//    const CellGeom    *cgL,*cgR;
//
//    for (face = fStart; face < fEnd; ++face) {
//      ierr = DMPlexGetLabelValue(dm, "ghost", face, &ghost);CHKERRQ(ierr);
//      if (ghost >= 0) continue;
//      ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);/*The support of a face is the cells (two cells)*/
//      ierr = DMPlexPointLocalRead(dmFace,face,facegeom,&fg);CHKERRQ(ierr);/*Read the data from "facegeom" for the point "face"*/
//      ierr = DMPlexPointLocalRead(dmCell,cells[0],cellgeom,&cgL);CHKERRQ(ierr);
//      ierr = DMPlexPointLocalRead(dmCell,cells[1],cellgeom,&cgR);CHKERRQ(ierr);
//
//      ierr = DMPlexPointGlobalRef(dm,cells[0],x,&fL);CHKERRQ(ierr); /*For the functions*/
//      ierr = DMPlexPointGlobalRef(dm,cells[1],x,&fR);CHKERRQ(ierr);
//
//      if (fL) ierr = ExactSolution(0.0, cgL->centroid, fg->normal, fL, fL, user);CHKERRQ(ierr);
//      if (fR) ierr = ExactSolution(0.0, cgR->centroid, fg->normal, fR, fR, user);CHKERRQ(ierr);
//
//    }
//  }
//
//  ierr = VecRestoreArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
//  ierr = VecRestoreArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
//  ierr = VecRestoreArrayRead(locX,&x);CHKERRQ(ierr);
//
//  PetscFunctionReturn(0);
//}

#undef __FUNCT__
#define __FUNCT__ "ComputeExactSolution"
PetscErrorCode ComputeExactSolution(DM dm, PetscReal time, Vec X, User user)
{
  DM                dmCell;
  const PetscScalar *cellgeom;
  PetscScalar       *x;
  PetscInt          cStart, cEnd, cEndInterior = user->cEndInterior, c;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetDM(user->cellgeom, &dmCell);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->cellgeom, &cellgeom);CHKERRQ(ierr);
  ierr = VecGetArray(X, &x);CHKERRQ(ierr);
  for (c = cStart; c < cEndInterior; ++c) {
    const CellGeom *cg;
    PetscScalar    *xc;

    ierr = DMPlexPointLocalRead(dmCell,c,cellgeom,&cg);CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dm,c,x,&xc);CHKERRQ(ierr);
    if (xc) {ierr = ExactSolution(time, cg->centroid, xc, user);CHKERRQ(ierr);}
  }
  ierr = VecRestoreArrayRead(user->cellgeom, &cellgeom);CHKERRQ(ierr);
  ierr = VecRestoreArray(X, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

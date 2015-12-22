#include <petscts.h>
#include <petscfv.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <petscblaslapack.h>

#include "AeroSim.h"

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
/**
 Reading in the mesh data and distribute it to each
 processor.
*/
PetscErrorCode CreateMesh(MPI_Comm comm, User user)
{
  PetscErrorCode ierr;
  PetscInt       dim;
  PetscMPIInt    rank;
  PetscViewer    viewer;
  PetscReal      startwtime = 0.0, endwtime=0.0;/*Wall clock time*/

  PetscFunctionBeginUser;

  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  ierr = DMPlexCreateExodusFromFile(comm, user->filename, PETSC_TRUE, &user->dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(user->dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMPlexGetDimension(user->dm, &dim);CHKERRQ(ierr);

  {
    DM dmDist;

    ierr = DMPlexSetAdjacencyUseCone(user->dm, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMPlexSetAdjacencyUseClosure(user->dm, PETSC_FALSE);CHKERRQ(ierr);
//    ierr = DMPlexDistribute(user->dm, "chaco", overlap, NULL, &dmDist);CHKERRQ(ierr);
    startwtime = MPI_Wtime();
    ierr = DMPlexDistribute(user->dm, "metis", user->overlap, NULL, &dmDist);CHKERRQ(ierr);
    endwtime = MPI_Wtime();
    ierr = PetscPrintf(comm,"Partition the mesh takes %f s\n",endwtime-startwtime);CHKERRQ(ierr);
    /*Why do I need the overlap here? What's this overlap for?*/
    if (dmDist) {
      ierr = DMDestroy(&user->dm);CHKERRQ(ierr);
      user->dm   = dmDist;
    }
  }
  ierr = DMSetFromOptions(user->dm);CHKERRQ(ierr);

  {
    DM gdm;

    ierr = DMPlexGetHeightStratum(user->dm, 0, NULL, &user->cEndInterior);CHKERRQ(ierr);
    /*Here the user->cEndInterior is the number of element on each processor*/
    ierr = DMPlexConstructGhostCells(user->dm, NULL, &user->numGhostCells, &gdm);CHKERRQ(ierr);
    /*Add a layer of artificial elements on the boundary of the physical domain
      Note that it does not add any ghost cells inner boundary between subdomains*/
 //   PetscPrintf(PETSC_COMM_SELF, "rank[%d], cEndInterior = %d numGhostCells = %d\n", rank, user->cEndInterior, user->numGhostCells);
    ierr = DMDestroy(&user->dm);CHKERRQ(ierr);
    user->dm   = gdm;
    ierr = DMViewFromOptions(user->dm, NULL, "-dm_view");CHKERRQ(ierr);
  }

  if (user->show_partition) {
    DM  dmCell;
    Vec partition;

//    ierr = OutputVTK(dm, "cellgeom.vtk", &viewer);CHKERRQ(ierr);
//    ierr = VecView(user->cellgeom, viewer);CHKERRQ(ierr);
//    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = CreatePartitionVec(user->dm, &dmCell, &partition);CHKERRQ(ierr);
    ierr = OutputVTK(dmCell, "partition.vtk", &viewer);CHKERRQ(ierr);
    ierr = VecView(partition, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&partition);CHKERRQ(ierr);
    ierr = DMDestroy(&dmCell);CHKERRQ(ierr);
  }

  /* Count number of fields and dofs */
  Physics        phys;
  phys = user->model->physics;

  for (phys->nfields=0,phys->dof=0; phys->field_desc[phys->nfields].name; phys->nfields++) {
    phys->dof += phys->field_desc[phys->nfields].dof;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetUpLocalSpace"
/*@
  SetUpLocalSpace - Set up the dofs of each element

@*/
PetscErrorCode SetUpLocalSpace(User user)
{
  PetscSection   stateSection;
  Physics        phys;
  DM             dm =  user->dm;
  PetscInt       dof = user->model->physics->dof, *cind, d, stateSize, cStart, cEnd, c, i;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  /* Since we use the cell-centered scheme, we just need to set dof at the center of the cell */
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &stateSection);CHKERRQ(ierr);
  phys = user->model->physics;
  ierr = PetscSectionSetNumFields(stateSection,phys->nfields);CHKERRQ(ierr);
  for (i=0; i<phys->nfields; i++) {
    ierr = PetscSectionSetFieldName(stateSection,i,phys->field_desc[i].name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(stateSection,i,phys->field_desc[i].dof);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetChart(stateSection, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    for (i=0; i<phys->nfields; i++) {
      ierr = PetscSectionSetFieldDof(stateSection,c,i,phys->field_desc[i].dof);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetDof(stateSection, c, dof);CHKERRQ(ierr);
  }

  for (c = user->cEndInterior; c < cEnd; ++c) {
    ierr = PetscSectionSetConstraintDof(stateSection, c, dof);CHKERRQ(ierr);
    /*This is for the Dirichlet boundary conditions. With this set, the Dirichlet boundary conditions
      will be deleted from the matrix*/
  }
  ierr = PetscSectionSetUp(stateSection);CHKERRQ(ierr);
  ierr = PetscMalloc(dof * sizeof(PetscInt), &cind);CHKERRQ(ierr);
  for (d = 0; d < dof; d++){
    cind[d] = d;
  }

  for (c = user->cEndInterior; c < cEnd; ++c) {
    ierr = PetscSectionSetConstraintIndices(stateSection, c, cind);CHKERRQ(ierr);
  }
    /* This function is to set which dofs are the dirichlet boundary. For example:
       for the NS equations, usually only the velocity (u, v, w) are set to be dirichlet
       boundary on the boundary and the pressure, density, and energy are free on the boundary.
       If the unknows on the point is ordered as (\rho, u, v, w, p, e), then the cind[] = (1, 2, 3)
       in the above function. And also in the function PetscSectionSetConstraintDof(stateSection, c, dof),
       you also need to change the "dof" to 3.
       Since all the ghost cell are artifical cells, so all the dofs should be known, which means:
       cind[0, ..., dof] = (0, ..., dof).
    */

//  { /*Set the Constraint for the velocity field, Do I need to do this? Not sure.*/
//    PetscSection subSection;
//    PetscInt     *cind1;
//    ierr = PetscSectionGetField(stateSection, 1, &subSection);CHKERRQ(ierr);
//    for (c = user->cEndInterior; c < cEnd; ++c) {
//      ierr = PetscSectionSetConstraintDof(subSection, c, DIM);CHKERRQ(ierr);
//    }
//    ierr = PetscMalloc(DIM * sizeof(PetscInt), &cind1);CHKERRQ(ierr);
//    ierr = PetscSectionSetUp(subSection);CHKERRQ(ierr);
//    for (d = 0; d < DIM; d++){
//      cind1[d] = d;
//    }
//    for (c = user->cEndInterior; c < cEnd; ++c) {
//      ierr = PetscSectionSetConstraintIndices(subSection, c, cind1);CHKERRQ(ierr);
//    }
//    ierr = PetscFree(cind1);CHKERRQ(ierr);
//  }

  ierr = PetscFree(cind);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(stateSection, &stateSize);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm,stateSection);CHKERRQ(ierr);

//  PetscSection globalSection;
//  ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);
//  ierr = PetscSectionView(stateSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); /*view each subsection seperately*/
//  ierr = PetscSectionView_ASCII(stateSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); /*view the total section together*/

  ierr = PetscSectionDestroy(&stateSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartitionVec"
PetscErrorCode CreatePartitionVec(DM dm, DM *dmCell, Vec *partition)
{
  PetscSF        sfPoint;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscSection   sectionCell;
  PetscReal    *part;
  PetscInt       cStart, cEnd, c;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMClone(dm, dmCell);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = DMSetPointSF(*dmCell, sfPoint);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(*dmCell, coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dmCell, coordinates);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionCell);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmCell, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionCell, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = PetscSectionSetDof(sectionCell, c, 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sectionCell);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(*dmCell, sectionCell);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionCell);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(*dmCell, partition);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)*partition, "partition");CHKERRQ(ierr);
  ierr = VecGetArray(*partition, &part);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscReal *p;

    ierr = DMPlexPointLocalRef(*dmCell, c, part, &p);CHKERRQ(ierr);
    p[0] = rank;
  }
  ierr = VecRestoreArray(*partition, &part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ConstructGeometryFVM"
/**
 Set up face data and cell data.
 And also setup the gradient of the variables
 at the center of the cell.
 */
PetscErrorCode ConstructGeometryFVM(Vec *facegeom, Vec *cellgeom, User user)
{
  DM             dmFace, dmCell;
  PetscSection   sectionFace, sectionCell;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscReal      minradius;
  PetscReal    *fgeom, *cgeom;
  PetscInt       dim, cStart, cEnd, c, fStart, fEnd, f;
  PetscErrorCode ierr;
  PetscMPIInt rank;

  PetscFunctionBeginUser;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

  ierr = DMPlexGetDimension(user->dm, &dim);CHKERRQ(ierr);
//  PetscPrintf(PETSC_COMM_SELF, "rank[%d], dim = %d \n", rank, dim);
  if (dim != DIM) SETERRQ2(PetscObjectComm((PetscObject)user->dm),PETSC_ERR_SUP,"No support for dim %D != DIM %D",dim,DIM);
  ierr = DMGetCoordinateSection(user->dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(user->dm, &coordinates);CHKERRQ(ierr);

  /* Make cell centroids and volumes */
  ierr = DMClone(user->dm, &dmCell);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(dmCell, coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmCell, coordinates);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)user->dm), &sectionCell);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(user->dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  /*In the setup of the Chart, we first set up the chart as the sum of the cell and vertexs
   and then interpolate it the faces. So after that, the Height(0) is the cell, Height(1) is
   the faces, and if one want to get the vertex, one can use the Depth(0)*/
//  PetscPrintf(PETSC_COMM_SELF, "rank[%d], First = %d, End = %d, Number of cells with ghost = %d\n", rank, cStart, cEnd, cEnd-cStart);
  ierr = PetscSectionSetChart(sectionCell, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = PetscSectionSetDof(sectionCell, c, sizeof(CellGeom)/sizeof(PetscReal));CHKERRQ(ierr);
  }
  /*The defination of Structure CellGeom is:
    typedef struct {
      PetscReal centroid[DIM];
      PetscReal volume;
    } CellGeom;
    So here the DOF is set as 2, one is for the centriod and the other one is for volume.
  */
  ierr = PetscSectionSetUp(sectionCell);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dmCell, sectionCell);CHKERRQ(ierr);/*Update the section of the dmCell*/
  ierr = PetscSectionDestroy(&sectionCell);CHKERRQ(ierr); /* relinquish our reference */

  ierr = DMCreateLocalVector(dmCell, cellgeom);CHKERRQ(ierr);
  ierr = VecGetArray(*cellgeom, &cgeom);CHKERRQ(ierr);
  for (c = cStart; c < user->cEndInterior; ++c) {/*Only go through the interior elements, no ghost cells included*/
    CellGeom *cg;


    ierr = DMPlexPointLocalRef(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
    ierr = PetscMemzero(cg,sizeof(*cg));CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFVM(dmCell, c, &cg->volume, cg->centroid, NULL);CHKERRQ(ierr);

//    PetscInt  depth;
//    ierr = DMPlexGetLabelValue(dm, "depth", c, &depth);CHKERRQ(ierr);
//    PetscPrintf(PETSC_COMM_SELF, "rank[%d], depth = %d\n", rank, depth);
  }
  /* Compute face normals and minimum cell radius */
  ierr = DMClone(user->dm, &dmFace);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)user->dm), &sectionFace);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(user->dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
//  PetscPrintf(PETSC_COMM_SELF, "rank[%d], First = %d, Number of faces = %d\n", rank, fStart, fEnd-fStart);
  ierr = PetscSectionSetChart(sectionFace, fStart, fEnd);CHKERRQ(ierr);
  for (f = fStart; f < fEnd; ++f) {
    ierr = PetscSectionSetDof(sectionFace, f, sizeof(FaceGeom)/sizeof(PetscReal));CHKERRQ(ierr);
  }
  /*
  typedef struct {
    PetscReal normal[DIM];              // Area-scaled normals
    PetscReal centroid[DIM];            // Location of centroid (quadrature point)
    PetscReal grad[2][DIM];             // Face contribution to gradient in left and right cell
  } FaceGeom;
  */
  ierr = PetscSectionSetUp(sectionFace);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dmFace, sectionFace);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionFace);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmFace, facegeom);CHKERRQ(ierr);
  ierr = VecGetArray(*facegeom, &fgeom);CHKERRQ(ierr);
  minradius = PETSC_MAX_REAL; /*The minimum distance of the cell centers and face centers. */
//  PetscInt nn = 0;
//  PetscPrintf(PETSC_COMM_SELF, "Number of Face = %d\n",fEnd-fStart);
  for (f = fStart; f < fEnd; ++f) {
    FaceGeom *fg;
    PetscInt  ghost;

    ierr = DMPlexGetLabelValue(user->dm, "ghost", f, &ghost);CHKERRQ(ierr);
    /*
    The ghost face is defined as follows: the shared face of two cells
    on the neighber processor that one cell has at least one shared face
    with the cells on this processor, and the other cell does not share
    any faces with cells in this processor.
    */
    if (ghost >= 0) continue;
    ierr = DMPlexPointLocalRef(dmFace, f, fgeom, &fg);CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFVM(user->dm, f, NULL, fg->centroid, fg->normal);CHKERRQ(ierr);

//    ierr = PetscPrintf(PETSC_COMM_SELF,"The face %d: (%f, %f, %f)'s normal (%f, %f, %f)\n", nn++, fg->centroid[0], fg->centroid[1],fg->centroid[2], fg->normal[0], fg->normal[1], fg->normal[2]);CHKERRQ(ierr);

    /* Flip face orientation if necessary to match ordering in support, and Update minimum radius */
    {
      CellGeom       *cL, *cR;
      const PetscInt *cells;
      PetscReal      *lcentroid, *rcentroid;
      PetscReal     v[3];
      PetscInt        d;

      ierr = DMPlexGetSupport(user->dm, f, &cells);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell, cells[0], cgeom, &cL);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell, cells[1], cgeom, &cR);CHKERRQ(ierr);
      lcentroid = cells[0] >= user->cEndInterior ? fg->centroid : cL->centroid;
      rcentroid = cells[1] >= user->cEndInterior ? fg->centroid : cR->centroid;
      /*The expression (x ? y : z) has the value of y if x is nonzero, z otherwise
        So the above equations mean that if the support cell is a inner cell, use
        the cell centroid, otherwise use the face centroid*/
      WaxpyD(dim, -1, lcentroid, rcentroid, v); /*v = rcentroid - lcentroid*/
      if (DotD(dim, fg->normal, v) < 0) {/*DotD is the inner product of two vectors*/
        for (d = 0; d < dim; ++d) fg->normal[d] = -fg->normal[d];
      }
      /*This makes the normal direction always from "left" to "right" (we treat the cell[0] as left and cell[1] as right.)*/
      if (DotD(dim, fg->normal, v) <= 0) {
#if DIM == 2
        SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %d could not be fixed, normal (%g,%g) v (%g,%g)", f, fg->normal[0], fg->normal[1], v[0], v[1]);
#elif DIM == 3
        SETERRQ7(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %d could not be fixed, normal (%g,%g,%g) v (%g,%g,%g)", f, fg->normal[0], fg->normal[1], fg->normal[2], v[0], v[1], v[2]);
#else
#  error DIM not supported
#endif
      }
      if (cells[0] < user->cEndInterior) {
        WaxpyD(dim, -1, fg->centroid, cL->centroid, v);
        minradius = PetscMin(minradius, NormD(dim, v));
      }
      if (cells[1] < user->cEndInterior) {
        WaxpyD(dim, -1, fg->centroid, cR->centroid, v);
        minradius = PetscMin(minradius, NormD(dim, v));
      }
    }
  }
  /* Compute centroids of ghost cells */
  for (c = user->cEndInterior; c < cEnd; ++c) {
    FaceGeom       *fg;
    const PetscInt *cone,    *support;
    PetscInt        coneSize, supportSize, s;

    ierr = DMPlexGetConeSize(dmCell, c, &coneSize);CHKERRQ(ierr);
    if (coneSize != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Ghost cell %d has cone size %d != 1", c, coneSize);
    ierr = DMPlexGetCone(dmCell, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dmCell, cone[0], &supportSize);CHKERRQ(ierr);
    if (supportSize != 2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %d has support size %d != 2", cone[0], supportSize);
    ierr = DMPlexGetSupport(dmCell, cone[0], &support);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRef(dmFace, cone[0], fgeom, &fg);CHKERRQ(ierr);
    for (s = 0; s < 2; ++s) {
      /* Reflect ghost centroid across plane of face */
      if (support[s] == c) { /*means that support[s] is the ghost cell*/
        const CellGeom *ci;
        CellGeom       *cg;
        PetscReal     c2f[3], a;

        ierr = DMPlexPointLocalRead(dmCell, support[(s+1)%2], cgeom, &ci);CHKERRQ(ierr);/* support[(s+1)%2] is the inner cell*/
        WaxpyD(dim, -1, ci->centroid, fg->centroid, c2f); /* inner cell to face centroid */
        a    = DotD(dim, c2f, fg->normal)/DotD(dim, fg->normal, fg->normal);
        /* a is the scalar prejection of c2f on the direction fg->normal */
        ierr = DMPlexPointLocalRef(dmCell, support[s], cgeom, &cg);CHKERRQ(ierr);
        WaxpyD(dim, 2*a, fg->normal, ci->centroid, cg->centroid);
        /*Caculate the centroid of the ghost cell. The ghost cell centroid is the reflect of
          the inner cell centriod on the face normal direction*/
        cg->volume = ci->volume;
      }
    }
  }
  if (user->reconstruct || !(user->Euler)) {
    PetscSection sectionGrad;
    ierr = BuildLeastSquares(user->dm, user->cEndInterior, dmFace, fgeom, dmCell, cgeom);CHKERRQ(ierr);
    ierr = DMClone(user->dm,&user->dmGrad);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)user->dm),&sectionGrad);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(sectionGrad,cStart,cEnd);CHKERRQ(ierr);
//    PetscPrintf(PETSC_COMM_SELF, "rank[%d], DOF = %d\n", rank, user->model->physics->dof);
    for (c=cStart; c<cEnd; c++) {
      ierr = PetscSectionSetDof(sectionGrad,c,user->model->physics->dof*DIM);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(sectionGrad);CHKERRQ(ierr);
    ierr = DMSetDefaultSection(user->dmGrad,sectionGrad);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&sectionGrad);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(*facegeom, &fgeom);CHKERRQ(ierr);
  ierr = VecRestoreArray(*cellgeom, &cgeom);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&minradius, &user->minradius, 1, MPIU_SCALAR, MPI_MIN, PetscObjectComm((PetscObject)user->dm));CHKERRQ(ierr);
  ierr = DMDestroy(&dmCell);CHKERRQ(ierr);
  ierr = DMDestroy(&dmFace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BuildLeastSquares"
/**
 Build least squares gradient reconstruction operators

      The gradient of u at the center of the cell should be the least square solution of the following system
           _                             _   _      _     _       _
          | x1 - x0    y1 - y0    z1 - z0 | |  gradx |   | u1 - u0 |
          | x2 - x0    y2 - y0    z2 - z0 | |  grady |   | u2 - u0 |
          | x3 - x0    y3 - y0    z3 - z0 | |_ gradz_| = | u3 - u0 |    (1)
          |_x4 - x0    y4 - y0    z4 - z0_|              |_u4 - u0_|

       So it can be splitted into four systems (based on the faces of the cell): The right hand side are the same
       and the left hand side are four vectors: (u1-u0, 0, 0, 0)^T,  (0, u2-u0, 0, 0)^T, (0, 0, u3-u0, 0)^T,
       (0, 0, 0, u4-u0)^T. And the solution of system (1) is the sum of the solution of these four systems.
*/
PetscErrorCode BuildLeastSquares(DM dm,PetscInt cEndInterior,DM dmFace,PetscReal *fgeom,DM dmCell,PetscReal *cgeom)
{
  PetscErrorCode ierr;
  PetscInt       c,cStart,cEnd,maxNumFaces,worksize;
  PetscReal    *B,*Binv,*work,*tau,**gref;

  PetscFunctionBeginUser;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(dm,&maxNumFaces,NULL);CHKERRQ(ierr);/* obtain the maximum of the cone size, that is the number of faces*/
  ierr = PseudoInverseGetWorkRequired(maxNumFaces,&worksize);CHKERRQ(ierr);
  ierr = PetscMalloc5(maxNumFaces*DIM,&B,worksize,&Binv,worksize,&work,maxNumFaces,&tau,maxNumFaces,&gref);CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_SELF, "maxNumFaces = %d, worksize = %d\n", maxNumFaces, worksize);
//  int number = 0;
  for (c=cStart; c<cEndInterior; c++) {
    const PetscInt *faces;
    PetscInt       numFaces,usedFaces,f,i,j;
    const CellGeom *cg;
    PetscBool      ghost;
    ierr = DMPlexGetConeSize(dm,c,&numFaces);CHKERRQ(ierr);
    if (numFaces < DIM) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell %D has only %D faces, not enough for gradient reconstruction",c,numFaces);
    ierr = DMPlexGetCone(dm,c,&faces);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell,c,cgeom,&cg);CHKERRQ(ierr);
    for (f=0,usedFaces=0; f<numFaces; f++) {
      const PetscInt *fcells;
      PetscInt       ncell,side;
      FaceGeom       *fg;
      const CellGeom *cg1;
      ierr = IsExteriorGhostFace(dm,faces[f],&ghost);CHKERRQ(ierr);
      if (ghost) continue;
      ierr  = DMPlexGetSupport(dm,faces[f],&fcells);CHKERRQ(ierr);
      side  = (c != fcells[0]); /* c is on left=0 or right=1 of face */
//      PetscPrintf(PETSC_COMM_SELF, "side = %d\n",side);
      ncell = fcells[!side];   /* the neighbor */
      ierr  = DMPlexPointLocalRef(dmFace,faces[f],fgeom,&fg);CHKERRQ(ierr);
      ierr  = DMPlexPointLocalRead(dmCell,ncell,cgeom,&cg1);CHKERRQ(ierr);
      for (j=0; j<DIM; j++) {
        B[j*numFaces+usedFaces] = cg1->centroid[j] - cg->centroid[j];
        if (PetscAbsScalar(B[j*numFaces+usedFaces]) < MYTOLERANCE) B[j*numFaces+usedFaces] = 0.0;
        //PetscPrintf(PETSC_COMM_SELF, " %g ", B[j*numFaces+usedFaces]);
      }
      //PetscPrintf(PETSC_COMM_SELF, "\n");
      /*   _                             _     _                 _
          | x1 - x0    y1 - y0    z1 - z0 |   | B[0]  B[4]  B[8]  |
          | x2 - x0    y2 - y0    z2 - z0 |   | B[1]  B[5]  B[9]  |
      B = | x3 - x0    y3 - y0    z3 - z0 | = | B[2]  B[6]  B[10] |
          |_x4 - x0    y4 - y0    z4 - z0_|   |_B[3]  B[7]  B[11]_|
      */
      gref[usedFaces++] = fg->grad[side];
      /* Gradient reconstruction term will go here and here fd->grad is not the acture gradient,
        it is just the least square inverse of the matrix B and one needs to multiply
        the difference (u1 - u0) etc to get the gradient.
        The defination of fg->grad: PetscReal grad[2][DIM]; // Face contribution to gradient in left and right cell
      */
    }
    //PetscPrintf(PETSC_COMM_SELF, "\n");
    if (!usedFaces) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Mesh contains isolated cell (no neighbors). Is it intentional?");
    /* Overwrites B with garbage, returns Binv in row-major format */
    if (0) {
      ierr = PseudoInverse(usedFaces,numFaces,DIM,B,Binv,tau,worksize,work);CHKERRQ(ierr);
      //ierr = PetscFVLeastSquaresPseudoInverse_Static(usedFaces,numFaces,DIM,B,Binv,tau,worksize,work);CHKERRQ(ierr);
    } else {
      ierr = PseudoInverseSVD(usedFaces,numFaces,DIM,B,Binv,tau,worksize,work);CHKERRQ(ierr);
      //ierr = PetscFVLeastSquaresPseudoInverseSVD_Static(usedFaces,numFaces,DIM,B,Binv,tau,worksize,work);CHKERRQ(ierr);
    }
    /*
      Binv[0,4,8] = X0, Binv[1,5,9] = X1, Binv[2,6,10] = X2, Binv[3,7,10] = X3, where
      X0,...,X3 are the solutions of the least square solution of AX-B = 0
      with the right hand side B = (b1, b2, b3, b4) = E (E is the 4 by 4 identity matrix).
    */
    for (f=0,i=0; f<numFaces; f++) {
      ierr = IsExteriorGhostFace(dm,faces[f],&ghost);CHKERRQ(ierr);
      if (ghost) continue;
      for (j=0; j<DIM; j++) {
        gref[i][j] = Binv[j*numFaces+i];
        //PetscPrintf(PETSC_COMM_SELF, " %g ", Binv[j*numFaces+i]);
      }
      i++;
      //PetscPrintf(PETSC_COMM_SELF, "\n");
    }
    //PetscPrintf(PETSC_COMM_SELF, "\n");
/*
       fg0->grad[0 or 1][0 to 2]: gref[0][0] = Binv[0]  gref[0][1] = Binv[4]  gref[0][2] = Binv[8]
       fg1->grad[0 or 1][0 to 2]: gref[1][0] = Binv[1]  gref[1][1] = Binv[5]  gref[1][2] = Binv[9]
       fg2->grad[0 or 1][0 to 2]: gref[2][0] = Binv[2]  gref[2][1] = Binv[6]  gref[2][2] = Binv[10]
       fg3->grad[0 or 1][0 to 2]: gref[3][0] = Binv[3]  gref[3][1] = Binv[7]  gref[3][2] = Binv[11]

       That's fg0->grad[0 or 1][0 to 2] = X0, fg1->grad[0 or 1][0 to 2] = X1,
              fg2->grad[0 or 1][0 to 2] = X2, fg3->grad[0 or 1][0 to 2] = X3.
*/

#if 0
      PetscReal grad[3] = {0,0,0};
      for (f=0; f<numFaces; f++) {
        const PetscInt *fcells;
        const CellGeom *cg1;
        const FaceGeom *fg;
        ierr = DMPlexGetSupport(dm,faces[f],&fcells);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dmFace,faces[f],fgeom,&fg);CHKERRQ(ierr);
        for (i=0; i<2; i++) {
          if (fcells[i] == c) continue;
          ierr = DMPlexPointLocalRead(dmCell,fcells[i],cgeom,&cg1);CHKERRQ(ierr);
          PetscReal du = cg1->centroid[0]*cg1->centroid[0] + 3*cg1->centroid[1] + 6*cg1->centroid[2] -
                          (cg->centroid[0]*cg->centroid[0]  + 3*cg->centroid[1]  + 6*cg->centroid[2]);
          grad[0] += fg->grad[!i][0] * du;
          grad[1] += fg->grad[!i][1] * du;
          grad[2] += fg->grad[!i][2] * du;
          //PetscPrintf(PETSC_COMM_SELF, "cell[%d], i %d, du %g, fg->grad (%g,%g,%g), grad (%g,%g,%g)\n",c,i,du,fg->grad[!i][0],fg->grad[!i][1], fg->grad[!i][2],grad[0],grad[1], grad[2]);
        }
      }
      PetscPrintf(PETSC_COMM_SELF, "cell[%d] coord (%g,%g,%g), grad (%g,%g,%g), error (%g,%g,%g)\n",c,cg->centroid[0], cg->centroid[1], cg->centroid[2], grad[0],grad[1], grad[2], PetscAbsScalar(2*cg->centroid[0]-grad[0]),PetscAbsScalar(3-grad[1]), PetscAbsScalar(6-grad[2]));
#endif
  }
  ierr = PetscFree5(B,Binv,work,tau,gref);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PseudoInverseGetWorkRequired"
PetscErrorCode PseudoInverseGetWorkRequired(PetscInt maxFaces,PetscInt *work)
{
  PetscInt m,n,nrhs,minwork;

  PetscFunctionBeginUser;
  m       = maxFaces;
  n       = DIM;
  nrhs    = maxFaces;
  minwork = 3*PetscMin(m,n) + PetscMax(2*PetscMin(m,n), PetscMax(PetscMax(m,n), nrhs)); /* required by LAPACK */
  *work   = 5*minwork;          /* We can afford to be extra generous */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IsExteriorGhostFace"
PetscErrorCode IsExteriorGhostFace(DM dm,PetscInt face,PetscBool *isghost)
{
  PetscErrorCode ierr;
  PetscInt       ghost,boundary;

  PetscFunctionBeginUser;
  *isghost = PETSC_FALSE;
  ierr = DMPlexGetLabelValue(dm, "ghost", face, &ghost);CHKERRQ(ierr);
  ierr = DMPlexGetLabelValue(dm, "Face Sets", face, &boundary);CHKERRQ(ierr);
  if (ghost >= 0 || boundary >= 0) *isghost = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PseudoInverseSVD"
/**
  Overwrites A. Can handle degenerate problems and m<n.
  The pseudoinverse is also called Moore–Penrose pseudoinverse,
  is a generalization of the inverse matrix for a rectangular matrix.
  The pseudoinverse of a matrix A is denoted as A^{+} which satisfies
  AA^{+}A = A, that is AA^{+} need not be the general identity matrix,
  but it maps all column vectors of A to themselves.
*/
PetscErrorCode PseudoInverseSVD(PetscInt m,PetscInt mstride,PetscInt n,PetscReal *A,PetscReal *Ainv,PetscReal *tau,PetscInt worksize,PetscReal *work)
{ /* ierr = PseudoInverseSVD(usedFaces,numFaces,DIM,B,Binv,tau,worksize,work);CHKERRQ(ierr);  */
  PetscBool      debug = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt       i,j,maxmn;
  PetscBLASInt   M,N,nrhs,lda,ldb,irank,ldwork,info;
  PetscReal    rcond,*tmpwork,*Brhs,*Aback;

  PetscFunctionBeginUser;
  if (debug) {
    ierr = PetscMalloc(m*n*sizeof(PetscReal),&Aback);CHKERRQ(ierr);
    ierr = PetscMemcpy(Aback,A,m*n*sizeof(PetscReal));CHKERRQ(ierr);
  }

  /* initialize to identity */
  tmpwork = Ainv;
  Brhs = work;
  maxmn = PetscMax(m,n);
  for (j=0; j<maxmn; j++) {
    for (i=0; i<maxmn; i++) Brhs[i + j*maxmn] = 1.0*(i == j);
  }

  ierr  = PetscBLASIntCast(m,&M);CHKERRQ(ierr);
  ierr  = PetscBLASIntCast(n,&N);CHKERRQ(ierr);
  nrhs  = M;
  ierr  = PetscBLASIntCast(mstride,&lda);CHKERRQ(ierr);
  ierr  = PetscBLASIntCast(maxmn,&ldb);CHKERRQ(ierr);
  ierr  = PetscBLASIntCast(worksize,&ldwork);CHKERRQ(ierr);
  rcond = -1;
  ierr  = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  LAPACKgelss_(&M,&N,&nrhs,A,&lda,Brhs,&ldb,tau,&rcond,&irank,tmpwork,&ldwork,&info);

/*
    Definition:
  ===========

       SUBROUTINE SGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK,
                          WORK, LWORK, INFO )

       .. Scalar Arguments ..
       INTEGER            INFO, LDA, LDB, LWORK, M, N, NRHS, RANK
       REAL               RCOND
       ..
       .. Array Arguments ..
       REAL               A( LDA, * ), B( LDB, * ), S( * ), WORK( * )
       ..


    Purpose:
  =============

  SGELSS computes the minimum norm solution to a real linear least
  squares problem:

  Minimize 2-norm(| b - A*x |).

  using the singular value decomposition (SVD) of A. A is an M-by-N
  matrix which may be rank-deficient.

  Several right hand side vectors b and solution vectors x can be
  handled in a single call; they are stored as the columns of the
  M-by-NRHS right hand side matrix B and the N-by-NRHS solution matrix
  X.

  The effective rank of A is determined by treating as zero those
  singular values which are less than RCOND times the largest singular
  value.
  \endverbatim
*
*  Arguments:
*  ==========
*
  \param[in] M
  \verbatim
           M is INTEGER
           The number of rows of the matrix A. M >= 0.
  \endverbatim

  \param[in] N
  \verbatim
           N is INTEGER
           The number of columns of the matrix A. N >= 0.
  \endverbatim

  \param[in] NRHS
  \verbatim
           NRHS is INTEGER
           The number of right hand sides, i.e., the number of columns
           of the matrices B and X. NRHS >= 0.
  \endverbatim

  \param[in,out] A
  \verbatim
           A is REAL array, dimension (LDA,N)
           On entry, the M-by-N matrix A.
           On exit, the first min(m,n) rows of A are overwritten with
           its right singular vectors, stored rowwise.
  \endverbatim

  \param[in] LDA
  \verbatim
           LDA is INTEGER
           The leading dimension of the array A.  LDA >= max(1,M).
  \endverbatim

  \param[in,out] B
  \verbatim
           B is REAL array, dimension (LDB,NRHS)
           On entry, the M-by-NRHS right hand side matrix B.
           On exit, B is overwritten by the N-by-NRHS solution
           matrix X.  If m >= n and RANK = n, the residual
           sum-of-squares for the solution in the i-th column is given
           by the sum of squares of elements n+1:m in that column.
  \endverbatim

  \param[in] LDB
  \verbatim
           LDB is INTEGER
           The leading dimension of the array B. LDB >= max(1,max(M,N)).
  \endverbatim

  \param[out] S
  \verbatim
           S is REAL array, dimension (min(M,N))
           The singular values of A in decreasing order.
           The condition number of A in the 2-norm = S(1)/S(min(m,n)).
  \endverbatim

  \param[in] RCOND
  \verbatim
           RCOND is REAL
           RCOND is used to determine the effective rank of A.
           Singular values S(i) <= RCOND*S(1) are treated as zero.
           If RCOND < 0, machine precision is used instead.
  \endverbatim

  \param[out] RANK
  \verbatim
           RANK is INTEGER
           The effective rank of A, i.e., the number of singular values
           which are greater than RCOND*S(1).
  \endverbatim

  \param[out] WORK
  \verbatim
           WORK is REAL array, dimension (MAX(1,LWORK))
           On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
  \endverbatim

  \param[in] LWORK
  \verbatim
           LWORK is INTEGER
           The dimension of the array WORK. LWORK >= 1, and also:
           LWORK >= 3*min(M,N) + max( 2*min(M,N), max(M,N), NRHS )
           For good performance, LWORK should generally be larger.

           If LWORK = -1, then a workspace query is assumed; the routine
           only calculates the optimal size of the WORK array, returns
           this value as the first entry of the WORK array, and no error
           message related to LWORK is issued by XERBLA.
  \endverbatim

  \param[out] INFO
  \verbatim
           INFO is INTEGER
           = 0:  successful exit
           < 0:  if INFO = -i, the i-th argument had an illegal value.
           > 0:  the algorithm for computing the SVD failed to converge;
                 if INFO = i, i off-diagonal elements of an intermediate
                 bidiagonal form did not converge to zero.
  \endverbatim
*
*/

  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xGELSS error");
  /* The following check should be turned into a diagnostic as soon as someone wants to do this intentionally */
  if (irank < PetscMin(M,N)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"WORNING!!! Rank deficient least squares fit, indicates an isolated cell with two colinear points, Check the function PseudoInverseSVD in SetupMeshes.c\n");CHKERRQ(ierr);
    //SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Rank deficient least squares fit, indicates an isolated cell with two colinear points");
  }
  /* Brhs shaped (M,nrhs) column-major coldim=mstride was overwritten by Ainv shaped (N,nrhs) column-major coldim=maxmn.
   * Here we transpose to (N,nrhs) row-major rowdim=mstride. */
  for (i=0; i<n; i++) {
    for (j=0; j<nrhs; j++) Ainv[i*mstride+j] = Brhs[i + j*maxmn];
  }
  /*
           _     _
          |  X0 * |
          |  X1 * |
   Brhs = |  X2 * |
          |_ X3 *_|4x4

  where X0,...,X3 (1 by 3 vectors) are the solutions of the least square problem
  with the right hand side B = E (E is the 4 by 4 identity matrix).

  A[0] = B[0], A[1] = B[4], A[2]  = B[8],   A[3]  = B[12],
  A[4] = B[1], A[5] = B[5], A[6]  = B[9],   A[7]  = B[13],
  A[8] = B[2], A[9] = B[6], A[10] = B[10],  A[11] = B[14],

  That's A[0,4,8] = X0, A[1,5,9] = X1, A[2,6,10] = X2, A[3,7,10] = X3.

  */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PseudoInverse"
/* Overwrites A. Can only handle full-rank problems with m>=n */
PetscErrorCode PseudoInverse(PetscInt m,PetscInt mstride,PetscInt n,PetscReal *A,PetscReal *Ainv,PetscReal *tau,PetscInt worksize,PetscReal *work)
{
  PetscBool      debug = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscBLASInt   M,N,K,lda,ldb,ldwork,info;
  PetscReal    *R,*Q,*Aback,Alpha;

  PetscFunctionBeginUser;
  if (debug) {
    ierr = PetscMalloc(m*n*sizeof(PetscReal),&Aback);CHKERRQ(ierr);
    ierr = PetscMemcpy(Aback,A,m*n*sizeof(PetscReal));CHKERRQ(ierr);
  }

  ierr = PetscBLASIntCast(m,&M);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&N);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(mstride,&lda);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(worksize,&ldwork);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  LAPACKgeqrf_(&M,&N,A,&lda,tau,work,&ldwork,&info);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xGEQRF error");
  R = A; /* Upper triangular part of A now contains R, the rest contains the elementary reflectors */

  /* Extract an explicit representation of Q */
  Q    = Ainv;
  ierr = PetscMemcpy(Q,A,mstride*n*sizeof(PetscReal));CHKERRQ(ierr);
  K    = N;                     /* full rank */
  LAPACKungqr_(&M,&N,&K,Q,&lda,tau,work,&ldwork,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xORGQR/xUNGQR error");

  /* Compute A^{-T} = (R^{-1} Q^T)^T = Q R^{-T} */
  Alpha = 1.0;
  ldb   = lda;
  BLAStrsm_("Right","Upper","ConjugateTranspose","NotUnitTriangular",&M,&N,&Alpha,R,&lda,Q,&ldb);
  /* Ainv is Q, overwritten with inverse */

  if (debug) {                      /* Check that pseudo-inverse worked */
    PetscReal Beta = 0.0;
    PetscBLASInt    ldc;
    K   = N;
    ldc = N;
    BLASgemm_("ConjugateTranspose","Normal",&N,&K,&M,&Alpha,Ainv,&lda,Aback,&ldb,&Beta,work,&ldc);
    ierr = PetscRealView(n*n,work,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = PetscFree(Aback);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscFVLeastSquaresPseudoInverse_Static"
/* Overwrites A. Can only handle full-rank problems with m>=n */
PetscErrorCode PetscFVLeastSquaresPseudoInverse_Static(PetscInt m,PetscInt mstride,PetscInt n,PetscScalar *A,PetscScalar *Ainv,PetscScalar *tau,PetscInt worksize,PetscScalar *work)
{
  PetscBool      debug = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscBLASInt   M,N,K,lda,ldb,ldwork,info;
  PetscScalar    *R,*Q,*Aback,Alpha;

  PetscFunctionBegin;
  if (debug) {
    ierr = PetscMalloc1(m*n,&Aback);CHKERRQ(ierr);
    ierr = PetscMemcpy(Aback,A,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  ierr = PetscBLASIntCast(m,&M);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&N);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(mstride,&lda);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(worksize,&ldwork);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  LAPACKgeqrf_(&M,&N,A,&lda,tau,work,&ldwork,&info);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xGEQRF error");
  R = A; /* Upper triangular part of A now contains R, the rest contains the elementary reflectors */

  /* Extract an explicit representation of Q */
  Q    = Ainv;
  ierr = PetscMemcpy(Q,A,mstride*n*sizeof(PetscScalar));CHKERRQ(ierr);
  K    = N;                     /* full rank */
  LAPACKungqr_(&M,&N,&K,Q,&lda,tau,work,&ldwork,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xORGQR/xUNGQR error");

  /* Compute A^{-T} = (R^{-1} Q^T)^T = Q R^{-T} */
  Alpha = 1.0;
  ldb   = lda;
  BLAStrsm_("Right","Upper","ConjugateTranspose","NotUnitTriangular",&M,&N,&Alpha,R,&lda,Q,&ldb);
  /* Ainv is Q, overwritten with inverse */

  if (debug) {                      /* Check that pseudo-inverse worked */
    PetscScalar  Beta = 0.0;
    PetscBLASInt ldc;
    K   = N;
    ldc = N;
    BLASgemm_("ConjugateTranspose","Normal",&N,&K,&M,&Alpha,Ainv,&lda,Aback,&ldb,&Beta,work,&ldc);
    ierr = PetscScalarView(n*n,work,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = PetscFree(Aback);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFVLeastSquaresPseudoInverseSVD_Static"
/* Overwrites A. Can handle degenerate problems and m<n. */
PetscErrorCode PetscFVLeastSquaresPseudoInverseSVD_Static(PetscInt m,PetscInt mstride,PetscInt n,PetscScalar *A,PetscScalar *Ainv,PetscScalar *tau,PetscInt worksize,PetscScalar *work)
{
  PetscBool      debug = PETSC_FALSE;
  PetscScalar   *Brhs, *Aback;
  PetscScalar   *tmpwork;
  PetscReal      rcond;
  PetscInt       i, j, maxmn;
  PetscBLASInt   M, N, nrhs, lda, ldb, irank, ldwork, info;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (debug) {
    ierr = PetscMalloc1(m*n,&Aback);CHKERRQ(ierr);
    ierr = PetscMemcpy(Aback,A,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  /* initialize to identity */
  tmpwork = Ainv;
  Brhs = work;
  maxmn = PetscMax(m,n);
  for (j=0; j<maxmn; j++) {
    for (i=0; i<maxmn; i++) Brhs[i + j*maxmn] = 1.0*(i == j);
  }

  ierr  = PetscBLASIntCast(m,&M);CHKERRQ(ierr);
  ierr  = PetscBLASIntCast(n,&N);CHKERRQ(ierr);
  nrhs  = M;
  ierr  = PetscBLASIntCast(mstride,&lda);CHKERRQ(ierr);
  ierr  = PetscBLASIntCast(maxmn,&ldb);CHKERRQ(ierr);
  ierr  = PetscBLASIntCast(worksize,&ldwork);CHKERRQ(ierr);
  rcond = -1;
  ierr  = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  if (tmpwork && rcond) rcond = 0.0; /* Get rid of compiler warning */
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "I don't think this makes sense for complex numbers");
#else
  LAPACKgelss_(&M,&N,&nrhs,A,&lda,Brhs,&ldb, (PetscReal *) tau,&rcond,&irank,tmpwork,&ldwork,&info);
#endif
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xGELSS error");
  /* The following check should be turned into a diagnostic as soon as someone wants to do this intentionally */
  if (irank < PetscMin(M,N)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Rank deficient least squares fit, indicates an isolated cell with two colinear points");

  /* Brhs shaped (M,nrhs) column-major coldim=mstride was overwritten by Ainv shaped (N,nrhs) column-major coldim=maxmn.
   * Here we transpose to (N,nrhs) row-major rowdim=mstride. */
  for (i=0; i<n; i++) {
    for (j=0; j<nrhs; j++) Ainv[i*mstride+j] = Brhs[i + j*maxmn];
  }
  PetscFunctionReturn(0);
}

#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include "tutorial_shared_path.h"
#include <igl/png/readPNG.h>

#include "LibiglSpinXForm.h"


int main(int argc, char *argv[])
{
   //// Dataset 1
   //std::string objName = TUTORIAL_SHARED_PATH "/Keenan_sphere.obj";
   //std::string imageName = TUTORIAL_SHARED_PATH "/bumpy.png";

   // Dataset 2
   std::string objName = TUTORIAL_SHARED_PATH "/capsule.obj";
   std::string imageName = TUTORIAL_SHARED_PATH "/spacemonkey.png";

  // Step 1:  Load a mesh in PLY format, this is only one that reads the UV with the 3D model
  Eigen::MatrixXd V, TC, N;
  Eigen::MatrixXi F, FTC, FN;
  //   V  double matrix of vertex positions  #V by 3
  //   TC  double matrix of texture coordinats #TC by 2
  //   N  double matrix of corner normals #N by 3
  //   F  #F list of face indices into vertex positions
  //   FTC  #F list of face indices into vertex texture coordinates
  //   FN  #F list of face indices into vertex normals
  igl::readOBJ(objName.c_str(), V, TC, N, F, FTC, FN);
  // Print the vertices, faces and UV matrices sizes
  std::cout << "Vertices: " << V.rows() << " X " << V.cols() << std::endl;
  std::cout << "Faces:    " << F.rows() << " X " << F.cols() << std::endl;
  std::cout << "Texture Coordinates: " << TC.rows() << " X " << TC.cols() << std::endl;
  std::cout << "Face to vertex texture coordinates: " << FTC.rows() << " X " << FTC.cols() << std::endl;

  // Step 2: Read the image, allocate temporary buffers, using builtin png for now
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R, G, B, Alpha;
  igl::png::readPNG(imageName, R, G, B, Alpha);
  std::cout << "Image Size:    " << R.rows() << " X " << R.cols() << std::endl;

  // Step 3 : Scaling the curvature change by a range gating on the image values acquired
  const double range = 5.;
  Eigen::VectorXd Rho(F.rows());
  int uv_count = 0;
  for (int i = 0; i < F.rows(); i++)// compute average value over the face
  {
	  Rho(i) = 0;
	 
	/*  std::cout << "Face - " << i << ":" << F.row(i) << std::endl;
	  std::cout << "Face index into vertex texture indexes " << FTC.row(i) << std::endl;*/
	  Eigen::Vector3i IndexesToTexCoordinates = FTC.row(i);

	  // compute average value over the face with 3 vertices, tet mesh would not work 
	  Eigen::Vector2d curUV_1 = TC.row(IndexesToTexCoordinates(0));
	  Eigen::Vector2d curUV_2 = TC.row(IndexesToTexCoordinates(1));
	  Eigen::Vector2d curUV_3 = TC.row(IndexesToTexCoordinates(2));

	  // Debug here it should match : double bilinearinterpVal = BilinearFilterSample(R, curUV_1.x() * R.rows(), curUV_1.y() * R.cols());
	  Rho(i) += BilinearFilterSample(R, curUV_1.x() * R.rows(), curUV_1.y() * R.cols()) / 3.f;
	  Rho(i) += BilinearFilterSample(R, curUV_2.x() * R.rows(), curUV_2.y() * R.cols()) / 3.f;
	  Rho(i) += BilinearFilterSample(R, curUV_3.x() * R.rows(), curUV_3.y() * R.cols()) / 3.f;

	  /*std::cout << "Face" << i << ":" << F.row(i)<< std::endl;
	  std::cout << "uv(i " << i << "," << 0 << ") =" << curUV_1 << std::endl;
	  std::cout << "uv(i " << i << "," << 1 << ") =" << curUV_2 << std::endl;
	  std::cout << "uv(i " << i << "," << 2 << ") =" << curUV_3 << std::endl;*/

	  // map value to [-range,range]
	  Rho(i) = (2.*(Rho(i) - .5)) * range;
	  //std::cout << "Rho( " << i << ") ="<< Rho(i) << std::endl;
  }
  
  // Step 4 :  Build the DiracOperator - Rho deformation matrix for Eigen value computations
  spinX::QuaternionMatrix E;
  std::vector<spinX::Quaternion> lambda;  // local similarity transformation (one value per vertex)
  lambda.resize(V.rows());
  E.resize(V.rows(), V.rows());

  BuildEigenValProblem(V, F, Rho, E); //Such problem can be mathematically expressed as a linear problem of the form $ Ax = b $ where $ x $ is the vector of m unknowns
  EigenSolver::solve(E, lambda);

  // Step 5:: Now solving possion problem
  spinX::QuaternionMatrix L; // Laplace matrix
  vector<spinX::Quaternion> omega; // divergence of target edge vectors
  vector<spinX::Quaternion> newVertices;
  L.resize(V.rows(), V.rows());
  omega.resize(V.rows());
  newVertices.resize(V.rows());

  BuildLaplacian(V, F, L);
  BuildOmega(V, F, lambda, omega);
  LinearSolver::solve(L, newVertices, omega);
  normalizeSolution(V.rows(), newVertices);
  
  // Gathering the deformed vertices locations - faces remain the same
  Eigen::MatrixXd V_deformed = Eigen::MatrixXd(V.rows(),3);
  for (int i = 0; i < V.rows(); i++)
  {
	  V_deformed.row(i) =  newVertices[i].im();
  }

  // Plot the mesh and register the callback
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V_deformed, F);
 
  // Launch the two data sets
  viewer.launch();
}

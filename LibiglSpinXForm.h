#pragma once
#include "igl/igl_inline.h"
#include <Eigen/Core>
#include <algorithm>
#include <Eigen/Sparse>

#include "Quaternion.h"
#include "QuaternionMatrix.h"
#include "LinearSolver.h"
#include "EigenSolver.h"

namespace igl
{
	template <typename DerivedV, typename DerivedF, typename DerivedN>
	IGL_INLINE void per_face_areas(
		const Eigen::MatrixBase<DerivedV>& V,
		const Eigen::MatrixBase<DerivedF>& F,
		Eigen::PlainObjectBase<DerivedN> & A);
}

template <typename DerivedV, typename DerivedF, typename DerivedN>
IGL_INLINE void igl::per_face_areas(
	const Eigen::MatrixBase<DerivedV>& V,
	const Eigen::MatrixBase<DerivedF>& F,
	Eigen::PlainObjectBase<DerivedN> & A
)
{
	A.resize(F.rows());
	// loop over faces
	int Frows = F.rows();
#pragma omp parallel for if (Frows>10000)
	for (int i = 0; i < Frows; i++)
	{
		const Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v1 = V.row(F(i, 1)) - V.row(F(i, 0));
		const Eigen::Matrix<typename DerivedV::Scalar, 1, 3> v2 = V.row(F(i, 2)) - V.row(F(i, 0));
		A(i) = 0.5*v1.cross(v2).norm(); // Per face area
	}
}


void clamp(const int& w, const int& h, int& x, int& y)
// clamps coordinates to range [0,w-1] x [0,h-1]
{
	x = std::max(0, std::min(w - 1, x));
	y = std::max(0, std::min(h - 1, y));
}

double BilinearFilterSample(Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& Image, double x, double y)
{
	double ax = x - floor(x);
	double ay = y - floor(y);
	double bx = 1. - ax;
	double by = 1. - ay;
	int x0 = (int)floor(x);
	int y0 = (int)floor(y);
	int x1 = x0 + 1;
	int y1 = y0 + 1;

	int w = Image.rows();
	int h = Image.cols();

	clamp(w, h, x0, y0);
	clamp(w, h, x1, y1);

	double val1 = Image(x0, y0) / 255.f;
	double val2 = Image(x0, y1) / 255.f;
	double val3 = Image(x1, y0) / 255.f;
	double val4 = Image(x1, y1) / 255.f;

	return by * (bx * val1 + ax * val3) +
		ay * (bx * val2 + ax * val4);
}

void BuildEigenValProblem(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& Rho, spinX::QuaternionMatrix& E)
{
	Eigen::VectorXd A;
	igl::per_face_areas(V, F, A);

	// Quaternion calculus makes the expressions more intuitive and easier to read, unlike what is done in MATLAB code

	// loop over faces
	int Frows = F.rows();
	for (int k = 0; k < Frows; k++)
	{
		double a = -1 / (4 * A(k));
		double b = Rho(k) / 6;
		double c = A(k) * Rho(k) * Rho(k) / 9.;

		// get vertex indices
		int I[3] =
		{
		   F(k,0),
		   F(k,1),
		   F(k,2)
		};

		// compute edges across from each vertex
		spinX::Quaternion e[3];
		for (int i = 0; i < 3; i++)
		{
			/*std::cout << "Vertex first" << V.row(I[(i + 2) % 3]) << std::endl;
			std::cout << "Vertex next" << V.row(I[(i + 1) % 3]) << std::endl;*/
			e[i] = spinX::Quaternion(Eigen::Vector3d(V.row(I[(i + 2) % 3])))               // Edges are computed using imaginary Quaternions, ImH
				- spinX::Quaternion(Eigen::Vector3d(V.row(I[(i + 1) % 3])));

			/*std::cout << "Computed Edge:" << e[i] << std::endl;*/
		}

		// increment matrix entry for each ordered pair of vertices
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
			{
				/*std::cout << "Before :" << E(I[i], I[j]) << std::endl;
				std::cout << "First term :" << a * e[i] * e[j] << std::endl;
				std::cout << "Second term :" << b * (e[j] - e[i]) << std::endl;
				std::cout << "Third term :" << c << std::endl;
				std::cout << " first and second and third" << a * e[i] * e[j] + b * (e[j] - e[i]) + c << std::endl;*/
				E(I[i], I[j]) += a * e[i] * e[j] + b * (e[j] - e[i]) + c;
				/*std::cout << "Product of quaternions : " << e[i] * e[j] << std::endl;*/
				//std::cout << "Index: " << I[i] << "," << I[j] << std::endl;
				//std::cout << "E matrix (Discrete Dirac Equation - deformation): " << E(I[i], I[j]) << std::endl;
			}
	}
}

void BuildLaplacian(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, spinX::QuaternionMatrix& L)
// builds the cotan-Laplace operator
{
	// allocate a sparse |V|x|V| matrix
	int nV = (int)V.rows();
	
	// visit each face
	for (size_t i = 0; i < F.rows(); i++)
	{
		// visit each triangle corner
		for (int j = 0; j < 3; j++)
		{
			// get vertex indices
			int k0 = F(i, (j + 0) % 3);
			int k1 = F(i, (j + 1) % 3);
			int k2 = F(i, (j + 2) % 3);

			// get vertex positions
			Eigen::Vector3d f0 = Eigen::Vector3d(V.row(k0));
			Eigen::Vector3d f1 = Eigen::Vector3d(V.row(k1));
			Eigen::Vector3d f2 = Eigen::Vector3d(V.row(k2));

			// compute cotangent of the angle at the current vertex
			// (equal to cosine over sine, which equals the dot
			// product over the norm of the cross product)
			Eigen::Vector3d u1 = f1 - f0;
			Eigen::Vector3d u2 = f2 - f0;
			double cotAlpha = (u1.dot(u2)) / (u1.cross(u2)).norm();

			// add contribution of this cotangent to the matrix
			L(k1, k2) -= cotAlpha / 2.;
			L(k2, k1) -= cotAlpha / 2.;
			L(k1, k1) += cotAlpha / 2.;
			L(k2, k2) += cotAlpha / 2.;

			/*std::cout << "1st Term = " << L(k1, k2) << std::endl;
			std::cout << "2nd Term = " << L(k1, k2) << std::endl;
			std::cout << "3rd Term = " << L(k1, k2) << std::endl;
			std::cout << "4th Term = " << L(k1, k2) << std::endl;*/
		}
	}
}

template <class T>
inline void removeMean(std::vector<T>& v)
{
	T mean = 0.;

	for (size_t i = 0; i < v.size(); i++)
	{
		mean += v[i];
	}

	mean /= (double)v.size();

	for (size_t i = 0; i < v.size(); i++)
	{
		v[i] -= mean;
	}
}

void BuildOmega(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<spinX::Quaternion>& lambda, vector<spinX::Quaternion>& omega)
{
	// clear omega
	for (size_t i = 0; i < omega.size(); i++)
	{
		omega[i] = 0.;
	}

	// visit each face
	for (size_t i = 0; i < F.rows(); i++)
	{
		// get indices of the vertices of this face
		int v[3] = { F(i,0),
					 F(i,1),
					 F(i,2) };

		// visit each edge
		for (int j = 0; j < 3; j++)
		{
			// get vertices
			Eigen::Vector3d f0_vec = V.row(v[(j + 0) % 3]);
			Eigen::Vector3d f1_vec = V.row(v[(j + 1) % 3]);
			Eigen::Vector3d f2_vec = V.row(v[(j + 2) % 3]);

			spinX::Quaternion f0(f0_vec);
			spinX::Quaternion f1(f1_vec);
			spinX::Quaternion f2(f2_vec);

			// determine orientation of this edge
			int a = v[(j + 1) % 3];
			int b = v[(j + 2) % 3];
			if (a > b)
			{
				swap(a, b);
			}

			// compute transformed edge vector
			spinX::Quaternion lambda1 = lambda[a];
			spinX::Quaternion lambda2 = lambda[b];
			spinX::Quaternion e = V.row(b) - V.row(a);
			spinX::Quaternion eTilde = (1. / 3.) * (~lambda1) * e * lambda1 +
				(1. / 6.) * (~lambda1) * e * lambda2 +
				(1. / 6.) * (~lambda2) * e * lambda1 +
				(1. / 3.) * (~lambda2) * e * lambda2;

			// compute cotangent of the angle opposite the current edge
			Eigen::Vector3d u1 = (f1 - f0).im();
			Eigen::Vector3d u2 = (f2 - f0).im();
			double cotAlpha = (u1.dot(u2)) / (u1.cross(u2)).norm();

			// add contribution of this edge to the divergence at its vertices
			omega[a] -= cotAlpha * eTilde / 2.;
			omega[b] += cotAlpha * eTilde / 2.;
		}
	}

	removeMean(omega);
}


void normalizeSolution(size_t numRows, vector<spinX::Quaternion>& newVertices)
{
	// center vertices around the origin
	removeMean(newVertices);

	// find the vertex with the largest norm
	double r = 0.;
	for (size_t i = 0; i < numRows; i++)
	{
		r = max(r, newVertices[i].norm2());
	}
	r = sqrt(r);

	// rescale so that vertices have norm at most one
	for (size_t i = 0; i < numRows; i++)
	{
		newVertices[i] /= r;
	}
}

//RhoLessDiracMat* op = new RhoLessDiracMat(A); 
  //Spectra::SymEigsSolver< double, Spectra::SMALLEST_ALGE, RhoLessDiracMat> eigs(&(*op), 1, 2000); // Construct eigen solver object, requesting the smallest 1 eigenvalues
  //eigs.init();
  //eigs.compute();                // Solving: Sparse linear system
  //if (eigs.info() == Spectra::SUCCESSFUL)
  //{
	 // Eigen::VectorXd evalues = eigs.eigenvalues();
	 // std::cout << "Smallest Eigen value is" << evalues << std::endl;
	 // std::vector<spinX::Quaternion> lamda;
	 // // Now we can solve
	 // //toImagQuat(evalues, lamda);
	 // delete op;
  //}

//class RhoLessDiracMat // D- Rho operator as defined in the paper
//{
//private:
//	SpMat A;
//public:
//
//	RhoLessDiracMat(const SpMat& _A)
//	{
//		A = _A;
//	}
//	int rows() { return A.rows(); }
//	int cols() { return A.cols(); }
//	// y_out = M * x_in
//	void perform_op(double *x_in, double *y_out)
//	{
//		Eigen::VectorXd X;
//		Eigen::VectorXd y = A * X;
//		y_out = y.data();
//	}
//};

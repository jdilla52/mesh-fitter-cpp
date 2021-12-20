#include <catch2/catch.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <3dMethods/pca.h>
using namespace Eigen;
using namespace std;

SCENARIO("icp", "[icp.h]")
{
  MatrixXd mat(4, 3);
  mat << 1, 1, 1, -2, -1, -1, 1.2, 1.2, 1.2, 21, 21, 21;
  auto j = Pca(mat);

  cout << "eigenValues:\n" << get<0>(j) << endl;
  cout << "eigen vectors: \n" << get<1>(j) << endl;
  // make some real test data
}

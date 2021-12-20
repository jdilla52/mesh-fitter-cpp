//
// Created by Jacques Perrault on 10/2/21.
//
#include <catch2/catch.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <3dMethods/pca.h>
using namespace Eigen;
using namespace std;

SCENARIO("pca", "[pca.h]")
{
  MatrixXd mat(4, 3);
  mat << 1, 1, 1, -2, -1, -1, 1.2, 1.2, 1.2, 21, 21, 21;
  auto j = Pca(mat);

  cout << "eigenValues:\n" << get<0>(j) << endl;
  cout << "eigen vectors: \n" << get<1>(j) << endl;
  // make some real test data
}

SCENARIO("findRigidTransform", "[pca.h]")
{
  MatrixXd source(2, 3);
  source << 1, 1, 1, -1, -1, -1;
  MatrixXd target(2, 3);
  target << -1, -1, -1, 2, 2, 2;
  auto out = findRigidTransform(source, target);
  cout << "rotation: \n" << get<0>(out) << endl;
  cout << "translation: \n" << get<1>(out) << endl;
}
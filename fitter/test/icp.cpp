#include <catch2/catch.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readSTL.h>
#include <igl/readOFF.h>
#include <methods_config.h>

using namespace Eigen;
using namespace std;

SCENARIO("icp", "[icp.h]")
{
  const Eigen::MatrixXd V = (Eigen::MatrixXd(8, 3) <<
                                                   0.0, 0.0, 0.0,
      0.0, 0.0, 1.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 1.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 1.0,
      1.0, 1.0, 0.0,
      1.0, 1.0, 1.0).finished();

  const Eigen::MatrixXi F = (Eigen::MatrixXi(12, 3) <<
                                                    1, 7, 5,
      1, 3, 7,
      1, 4, 3,
      1, 2, 4,
      3, 8, 7,
      3, 4, 8,
      5, 7, 8,
      5, 8, 6,
      1, 5, 6,
      1, 6, 2,
      2, 6, 8,
      2, 8, 4).finished().array() - 1;

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.data().set_face_based(true);
  viewer.launch();
}

SCENARIO("midsole icp", "[icp.h]")
{
  Eigen::MatrixXd VA;
  Eigen::MatrixXi FA;
  const std::string dir = SOURCE_DIR;
  igl::readOFF(dir + "/test/test_data/test_11D.off", VA, FA);

  Eigen::MatrixXd VB;
  Eigen::MatrixXi FB;
  const std::string dir2 = SOURCE_DIR;
  igl::readOFF(dir2 + "/test/test_data/test_11D_rotated.off", VB, FB);

  // merge meshes for viewer
  // Concatenate (VA,FA) and (VB,FB) into (V,F)
  Eigen::MatrixXd V(VA.rows()+VB.rows(),VA.cols());
  V<<VA,VB;
  Eigen::MatrixXi F(FA.rows()+FB.rows(),FA.cols());
  F<<FA,(FB.array()+VA.rows());

  // blue color for faces of first mesh, orange for second
  Eigen::MatrixXd C(F.rows(),3);
  C<<
   Eigen::RowVector3d(0.2,0.3,0.8).replicate(FA.rows(),1),
      Eigen::RowVector3d(1.0,0.7,0.2).replicate(FB.rows(),1);

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.data().set_colors(C);
  viewer.data().set_face_based(true);
  viewer.launch();
}
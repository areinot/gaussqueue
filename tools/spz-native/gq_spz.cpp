// gq_spz: convert a Gaussian-splat PLY to SPZ (or to another PLY) with Niantic's
// reference library, natively, so there's no WebAssembly memory ceiling.
//
//   gq_spz <in.ply> <out.spz|out.ply> [--spz-version 3|4] [--flip]
//          [--origin X Y Z] [--yaw DEGREES] [--keep FRACTION]
//
// The output format follows the extension: .spz is packed SPZ, .ply is a float PLY.
// --spz-version 3  classic gzip-wrapped SPZ (what most apps read); 4 = zstd (library default)
// --flip           rotate 180 degrees about X (incl. spherical harmonics) so captures that
//                  are +Y-up (SplatKing/COLMAP) are the right way up in -Y-up viewers.
// --origin X Y Z   then move the point (X, Y, Z) to the origin (coordinates after --flip)
// --yaw DEGREES    then rotate about the Y (up) axis, including spherical harmonics
// --keep FRACTION keep only this fraction of the splats, the most important first: opacity
//                  times (volume / 90th-percentile volume, capped at 1)^0.1. That's LightGaussian's
//                  score without the per-view hit counts, so it needs no photos or GPU
//                  (a quick prune for mid-training snapshots; the real pruning stage is gs/prune.py)
// --self-test      check the spherical-harmonics rotation against the library's exact flip
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "cc/load-spz.h"

namespace {

// Real SH basis for bands 1..3 in the 3DGS/gsplat ordering and signs. Any consistent basis
// works for building rotation matrices; this one matches how the coefficients are stored.
void shBasis(int band, double x, double y, double z, double *out) {
  const double xx = x * x, yy = y * y, zz = z * z;
  if (band == 1) {
    const double c = 0.4886025119029199;
    out[0] = -c * y; out[1] = c * z; out[2] = -c * x;
  } else if (band == 2) {
    out[0] = 1.0925484305920792 * x * y;
    out[1] = -1.0925484305920792 * y * z;
    out[2] = 0.31539156525252005 * (2 * zz - xx - yy);
    out[3] = -1.0925484305920792 * x * z;
    out[4] = 0.5462742152960396 * (xx - yy);
  } else {
    out[0] = -0.5900435899266435 * y * (3 * xx - yy);
    out[1] = 2.890611442640554 * x * y * z;
    out[2] = -0.4570457994644658 * y * (4 * zz - xx - yy);
    out[3] = 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy);
    out[4] = -0.4570457994644658 * x * (4 * zz - xx - yy);
    out[5] = 1.445305721320277 * z * (xx - yy);
    out[6] = -0.5900435899266435 * x * (xx - 3 * yy);
  }
}

// Solve (A^T A) M = A^T B for the (n x n) matrix M by Gauss-Jordan elimination.
std::vector<double> leastSquares(const std::vector<double> &A, const std::vector<double> &B, int rows, int n) {
  std::vector<double> ata(n * n, 0.0), atb(n * n, 0.0);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        ata[i * n + j] += A[r * n + i] * A[r * n + j];
        atb[i * n + j] += A[r * n + i] * B[r * n + j];
      }
  for (int col = 0; col < n; ++col) {
    int piv = col;
    for (int r = col + 1; r < n; ++r)
      if (std::fabs(ata[r * n + col]) > std::fabs(ata[piv * n + col])) piv = r;
    for (int j = 0; j < n; ++j) {
      std::swap(ata[col * n + j], ata[piv * n + j]);
      std::swap(atb[col * n + j], atb[piv * n + j]);
    }
    const double d = ata[col * n + col];
    for (int j = 0; j < n; ++j) { ata[col * n + j] /= d; atb[col * n + j] /= d; }
    for (int r = 0; r < n; ++r) {
      if (r == col) continue;
      const double f = ata[r * n + col];
      for (int j = 0; j < n; ++j) { ata[r * n + j] -= f * ata[col * n + j]; atb[r * n + j] -= f * atb[col * n + j]; }
    }
  }
  return atb;   // M, row-major: M[i][j]
}

// Matrix M (n x n, row-major) for one SH band such that, for coefficients c of f, M c are the
// coefficients of f rotated by R (row-major 3x3): f'(d) = f(R^T d).
std::vector<double> shRotation(int band, const double R[9]) {
  const int n = 2 * band + 1, rows = 64;
  std::vector<double> A(rows * n), B(rows * n);
  for (int k = 0; k < rows; ++k) {   // Fibonacci directions
    const double t = (k + 0.5) / rows, phi = std::acos(1 - 2 * t), th = M_PI * (1 + std::sqrt(5.0)) * (k + 0.5);
    const double d[3] = {std::cos(th) * std::sin(phi), std::sin(th) * std::sin(phi), std::cos(phi)};
    const double rd[3] = {R[0] * d[0] + R[3] * d[1] + R[6] * d[2],   // R^T d
                          R[1] * d[0] + R[4] * d[1] + R[7] * d[2],
                          R[2] * d[0] + R[5] * d[1] + R[8] * d[2]};
    shBasis(band, d[0], d[1], d[2], &A[k * n]);
    shBasis(band, rd[0], rd[1], rd[2], &B[k * n]);
  }
  // A c' = B c  ->  c' = M c with M = (A^T A)^-1 A^T B
  return leastSquares(A, B, rows, n);
}

void rotateSh(spz::GaussianCloud &cloud, const double R[9]) {
  if (cloud.shDegree < 1 || cloud.numPoints == 0) return;
  const int perPoint = static_cast<int>(cloud.sh.size() / 3 / cloud.numPoints);   // coefficients
  std::vector<std::vector<double>> M;
  for (int l = 1; l <= std::min(cloud.shDegree, 3); ++l) M.push_back(shRotation(l, R));
  std::vector<double> tmp(7);
  for (int p = 0; p < cloud.numPoints; ++p) {
    float *c = &cloud.sh[static_cast<size_t>(p) * perPoint * 3];
    int start = 0;
    for (int l = 1; l <= std::min(cloud.shDegree, 3); ++l) {
      const int n = 2 * l + 1;
      const auto &m = M[l - 1];
      for (int ch = 0; ch < 3; ++ch) {
        for (int i = 0; i < n; ++i) {
          double s = 0;
          for (int j = 0; j < n; ++j) s += m[i * n + j] * c[(start + j) * 3 + ch];
          tmp[i] = s;
        }
        for (int i = 0; i < n; ++i) c[(start + i) * 3 + ch] = static_cast<float>(tmp[i]);
      }
      start += n;
    }
  }
}

// Rigid transform: p' = R (p - origin); rotations (x, y, z, w) are pre-multiplied by R.
void transform(spz::GaussianCloud &cloud, const double origin[3], double yawDeg) {
  const double a = yawDeg * M_PI / 180.0, c = std::cos(a), s = std::sin(a);
  const double R[9] = {c, 0, s, 0, 1, 0, -s, 0, c};
  const double qw = std::cos(a / 2), qy = std::sin(a / 2);   // quaternion of R (about +Y)
  for (int i = 0; i < cloud.numPoints; ++i) {
    float *p = &cloud.positions[3 * i];
    const double x = p[0] - origin[0], y = p[1] - origin[1], z = p[2] - origin[2];
    p[0] = static_cast<float>(R[0] * x + R[1] * y + R[2] * z);
    p[1] = static_cast<float>(R[3] * x + R[4] * y + R[5] * z);
    p[2] = static_cast<float>(R[6] * x + R[7] * y + R[8] * z);
    float *q = &cloud.rotations[4 * i];   // x, y, z, w
    const double x2 = q[0], y2 = q[1], z2 = q[2], w2 = q[3];
    // (0, qy, 0, qw) * (x2, y2, z2, w2)
    q[0] = static_cast<float>(qw * x2 + qy * z2);
    q[1] = static_cast<float>(qw * y2 + qy * w2);
    q[2] = static_cast<float>(qw * z2 - qy * x2);
    q[3] = static_cast<float>(qw * w2 - qy * y2);
  }
  if (yawDeg != 0.0) rotateSh(cloud, R);
}

int selfTest() {
  // Generic SH rotation by 180 degrees about X must match the library's exact sign flip.
  spz::GaussianCloud a;
  a.numPoints = 1; a.shDegree = 3;
  a.positions = {0, 0, 0}; a.scales = {0, 0, 0}; a.rotations = {0, 0, 0, 1}; a.alphas = {0}; a.colors = {0, 0, 0};
  for (int i = 0; i < 45; ++i) a.sh.push_back(std::sin(1.7f * i + 0.3f));
  spz::GaussianCloud b = a;
  b.rotate180DegAboutX();
  const double Rx[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};
  rotateSh(a, Rx);
  double err = 0;
  for (int i = 0; i < 45; ++i) err = std::max(err, std::fabs(double(a.sh[i]) - b.sh[i]));
  // Yaw by +90 then -90 must round-trip.
  spz::GaussianCloud c = b;
  const double o[3] = {0, 0, 0};
  transform(c, o, 90); transform(c, o, -90);
  double err2 = 0;
  for (int i = 0; i < 45; ++i) err2 = std::max(err2, std::fabs(double(c.sh[i]) - b.sh[i]));
  std::cout << "SH rotation vs exact flip: max error " << err << "; yaw round trip: " << err2 << "\n";
  return (err < 1e-4 && err2 < 1e-4) ? 0 : 1;
}

// Keep the `keep` fraction of splats with the highest opacity * normalized-volume^0.1.
void prune(spz::GaussianCloud &c, double keep) {
  const int n = c.numPoints;
  const int target = std::max(1, static_cast<int>(std::lround(n * keep)));
  if (target >= n) return;
  std::vector<float> vol(n), score(n);
  for (int i = 0; i < n; ++i)
    vol[i] = std::exp(c.scales[3 * i] + c.scales[3 * i + 1] + c.scales[3 * i + 2]);
  std::vector<float> tmp(vol);
  auto p90 = tmp.begin() + static_cast<long>(0.9 * (n - 1));
  std::nth_element(tmp.begin(), p90, tmp.end());
  const double ref = std::max(*p90, 1e-30f);
  for (int i = 0; i < n; ++i) {
    const double opacity = 1.0 / (1.0 + std::exp(-c.alphas[i]));
    score[i] = static_cast<float>(opacity * std::pow(std::min(vol[i] / ref, 1.0), 0.1));
  }
  tmp = score;
  auto cut = tmp.begin() + (n - target);
  std::nth_element(tmp.begin(), cut, tmp.end());
  const float threshold = *cut;
  int above = 0;
  for (float s : score) above += s > threshold;
  int ties = target - above;   // splats exactly at the threshold, to fill up to target
  const int shPer = static_cast<int>(c.sh.size() / n);
  int j = 0;
  for (int i = 0; i < n; ++i) {
    if (!(score[i] > threshold || (score[i] == threshold && ties-- > 0))) continue;
    for (int k = 0; k < 3; ++k) {
      c.positions[3 * j + k] = c.positions[3 * i + k];
      c.scales[3 * j + k] = c.scales[3 * i + k];
      c.colors[3 * j + k] = c.colors[3 * i + k];
    }
    for (int k = 0; k < 4; ++k) c.rotations[4 * j + k] = c.rotations[4 * i + k];
    c.alphas[j] = c.alphas[i];
    std::copy_n(c.sh.begin() + static_cast<long>(i) * shPer, shPer, c.sh.begin() + static_cast<long>(j) * shPer);
    ++j;
  }
  c.numPoints = j;
  c.positions.resize(3 * j);
  c.scales.resize(3 * j);
  c.colors.resize(3 * j);
  c.rotations.resize(4 * j);
  c.alphas.resize(j);
  c.sh.resize(static_cast<size_t>(j) * shPer);
}

}  // namespace

int main(int argc, char *argv[]) {
  if (argc >= 2 && std::strcmp(argv[1], "--self-test") == 0) return selfTest();
  if (argc < 3) {
    std::cerr << "usage: gq_spz <in.ply> <out.spz|out.ply> [--spz-version 3|4] [--flip] "
                 "[--origin X Y Z] [--yaw DEGREES] [--keep FRACTION]\n";
    return 2;
  }
  int version = 3;
  bool flip = false, move = false;
  double origin[3] = {0, 0, 0}, yaw = 0, keep = 1;
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--flip") == 0) {
      flip = true;
    } else if (std::strcmp(argv[i], "--spz-version") == 0 && i + 1 < argc) {
      version = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--origin") == 0 && i + 3 < argc) {
      for (int k = 0; k < 3; ++k) origin[k] = std::atof(argv[++i]);
      move = true;
    } else if (std::strcmp(argv[i], "--yaw") == 0 && i + 1 < argc) {
      yaw = std::atof(argv[++i]);
      move = true;
    } else if (std::strcmp(argv[i], "--keep") == 0 && i + 1 < argc) {
      keep = std::atof(argv[++i]);
    } else {
      std::cerr << "unknown option: " << argv[i] << "\n";
      return 2;
    }
  }
  if (!(keep > 0 && keep <= 1)) {
    std::cerr << "--keep must be a fraction in (0, 1]\n";
    return 2;
  }
  if (version != 3 && version != 4) {
    std::cerr << "--spz-version must be 3 or 4\n";
    return 2;
  }
  try {
    spz::UnpackOptions unpack;
    spz::GaussianCloud cloud = spz::loadSplatFromPly(argv[1], unpack);
    if (cloud.numPoints == 0) {
      std::cerr << "no splats in " << argv[1] << "\n";
      return 1;
    }
    const int before = cloud.numPoints;
    if (keep < 1) prune(cloud, keep);
    if (flip) cloud.rotate180DegAboutX();
    if (move) transform(cloud, origin, yaw);
    spz::PackOptions pack;
    pack.version = static_cast<uint32_t>(version);
    const std::string out = argv[2];
    const bool to_ply = out.size() >= 4 && out.compare(out.size() - 4, 4, ".ply") == 0;
    const bool ok = to_ply ? spz::saveSplatToPly(cloud, pack, out) : spz::saveSpz(cloud, pack, out);
    if (!ok) {
      std::cerr << "failed to write " << out << "\n";
      return 1;
    }
    std::cout << cloud.numPoints << " splats, SH degree " << cloud.shDegree
              << (to_ply ? std::string(", PLY") : ", SPZ v" + std::to_string(version))
              << (keep < 1 ? ", pruned from " + std::to_string(before) : std::string())
              << (flip ? ", flipped" : "") << (move ? ", start view applied" : "") << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

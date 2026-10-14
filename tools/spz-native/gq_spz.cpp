// gq_spz: convert a Gaussian-splat PLY to SPZ (or to another PLY) with Niantic's
// reference library, natively, so there's no WebAssembly memory ceiling.
//
//   gq_spz <in.ply> <out.spz|out.ply> [--spz-version 3|4] [--flip]
//
// The output format follows the extension: .spz is packed SPZ, .ply is a float PLY.
// --spz-version 3  classic gzip-wrapped SPZ (what most apps read); 4 = zstd (library default)
// --flip           rotate 180 degrees about X (incl. spherical harmonics) so captures that
//                  are +Y-up (SplatKing/COLMAP) are the right way up in -Y-up viewers.
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "cc/load-spz.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage: gq_spz <in.ply> <out.spz|out.ply> [--spz-version 3|4] [--flip]\n";
    return 2;
  }
  int version = 3;
  bool flip = false;
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--flip") == 0) {
      flip = true;
    } else if (std::strcmp(argv[i], "--spz-version") == 0 && i + 1 < argc) {
      version = std::atoi(argv[++i]);
    } else {
      std::cerr << "unknown option: " << argv[i] << "\n";
      return 2;
    }
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
    if (flip) cloud.rotate180DegAboutX();
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
              << (flip ? ", flipped" : "") << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

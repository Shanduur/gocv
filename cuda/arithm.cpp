#include "../core.h"
#include "arithm.h"
#include <string.h>

void GpuAbs(GpuMat src, GpuMat dst, Stream s) {
    cv::cuda::abs(*src, *dst, s);
}

void GpuThreshold(GpuMat src, GpuMat dst, double thresh, double maxval, int typ, Stream s) {
    cv::cuda::threshold(*src, *dst, thresh, maxval, typ, s);
}

void GpuFlip(GpuMat src, GpuMat dst, int flipCode, Stream s) {
    cv::cuda::flip(*src, *dst, flipCode, s);
}

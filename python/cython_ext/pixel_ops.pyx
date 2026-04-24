# cython: language_level=3, boundscheck=False, wraparound=False
# ApexVision-Core — Pixel Operations (Cython)
import numpy as np
cimport numpy as np

def normalize_image(np.ndarray[np.uint8_t, ndim=3] img):
    """Normalize uint8 BGR image to float32 [0,1]"""
    cdef int h = img.shape[0], w = img.shape[1], c = img.shape[2]
    out = np.empty((h, w, c), dtype=np.float32)
    cdef float[:,:,:] out_v = out
    cdef unsigned char[:,:,:] in_v = img
    cdef int i, j, k
    for i in range(h):
        for j in range(w):
            for k in range(c):
                out_v[i,j,k] = in_v[i,j,k] / 255.0
    return out

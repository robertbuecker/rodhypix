// Standalone NumPy wrapper for dxtbx's TY6 decompression routine.
//
// The decompression logic is adapted from dxtbx/src/dxtbx/boost_python/compression.cc
// and retains the original BSD-3-Clause licensing terms from dxtbx.

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cstddef>
#include <cstdint>

namespace {

inline std::uint32_t read_uint32_from_bytearray(const char *buf) {
  return static_cast<unsigned char>(buf[0])
         | (static_cast<unsigned char>(buf[1]) << 8)
         | (static_cast<unsigned char>(buf[2]) << 16)
         | (static_cast<unsigned char>(buf[3]) << 24);
}

inline std::uint16_t read_uint16_from_bytearray(const char *buf) {
  return static_cast<unsigned char>(buf[0])
         | (static_cast<unsigned char>(buf[1]) << 8);
}

void rod_TY6_decompress(std::int32_t *const ret,
                        const char *const buf_data,
                        const char *const buf_offsets,
                        const int slow,
                        const int fast) {
  const std::size_t BLOCKSIZE = 8;
  const signed int SHORT_OVERFLOW = 127;
  const signed int LONG_OVERFLOW = 128;

  const std::size_t nblock = (fast - 1) / (BLOCKSIZE * 2);
  const std::size_t nrest = (fast - 1) % (BLOCKSIZE * 2);

  for (std::size_t iy = 0; iy < static_cast<std::size_t>(slow); iy++) {
    std::size_t ipos = read_uint32_from_bytearray(buf_offsets + iy * sizeof(std::uint32_t));
    std::size_t opos = static_cast<std::size_t>(fast) * iy;

    int firstpx = static_cast<unsigned char>(buf_data[ipos++]) - 127;
    if (firstpx == LONG_OVERFLOW) {
      firstpx = static_cast<signed int>(read_uint32_from_bytearray(buf_data + ipos));
      ipos += 4;
    } else if (firstpx == SHORT_OVERFLOW) {
      firstpx = static_cast<signed short>(read_uint16_from_bytearray(buf_data + ipos));
      ipos += 2;
    }
    ret[opos++] = firstpx;

    for (std::size_t k = 0; k < nblock; k++) {
      const std::size_t bittypes = static_cast<unsigned char>(buf_data[ipos++]);
      const std::size_t nbits[2] = {bittypes & 15, (bittypes >> 4) & 15};

      for (int i = 0; i < 2; i++) {
        const std::size_t nbit = nbits[i];

        int zero_at = 0;
        if (nbit > 1) {
          zero_at = (1 << (nbit - 1)) - 1;
        }

        unsigned long long v = 0;
        for (std::size_t j = 0; j < nbit; j++) {
          v |= static_cast<unsigned long long>(static_cast<unsigned char>(buf_data[ipos++]))
               << (BLOCKSIZE * j);
        }

        const unsigned long long mask = (1ULL << nbit) - 1;
        for (std::size_t j = 0; j < BLOCKSIZE; j++) {
          ret[opos++] = static_cast<std::int32_t>(((v >> (nbit * j)) & mask) - zero_at);
        }
      }

      for (std::size_t i = opos - 2 * BLOCKSIZE; i < opos; i++) {
        int offset = ret[i];

        if (offset == LONG_OVERFLOW) {
          offset = static_cast<signed int>(read_uint32_from_bytearray(buf_data + ipos));
          ipos += 4;
        } else if (offset == SHORT_OVERFLOW) {
          offset = static_cast<signed short>(read_uint16_from_bytearray(buf_data + ipos));
          ipos += 2;
        }

        ret[i] = static_cast<std::int32_t>(offset + ret[i - 1]);
      }
    }

    for (std::size_t i = 0; i < nrest; i++) {
      int offset = static_cast<unsigned char>(buf_data[ipos++]) - 127;

      if (offset == LONG_OVERFLOW) {
        offset = static_cast<signed int>(read_uint32_from_bytearray(buf_data + ipos));
        ipos += 4;
      } else if (offset == SHORT_OVERFLOW) {
        offset = static_cast<signed short>(read_uint16_from_bytearray(buf_data + ipos));
        ipos += 2;
      }

      ret[opos] = static_cast<std::int32_t>(ret[opos - 1] + offset);
      opos++;
    }
  }
}

PyObject *decode_ty6_image(PyObject *, PyObject *args) {
  PyObject *linedata_obj = nullptr;
  PyObject *offsets_obj = nullptr;
  int ny = 0;
  int nx = 0;

  if (!PyArg_ParseTuple(args, "OOii", &linedata_obj, &offsets_obj, &ny, &nx)) {
    return nullptr;
  }

  PyArrayObject *linedata = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(linedata_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY));
  if (linedata == nullptr) {
    return nullptr;
  }

  PyArrayObject *offsets = reinterpret_cast<PyArrayObject *>(
      PyArray_FROM_OTF(offsets_obj, NPY_UINT32, NPY_ARRAY_IN_ARRAY));
  if (offsets == nullptr) {
    Py_DECREF(linedata);
    return nullptr;
  }

  if (PyArray_NDIM(linedata) != 1 || PyArray_NDIM(offsets) != 1) {
    Py_DECREF(offsets);
    Py_DECREF(linedata);
    PyErr_SetString(PyExc_ValueError, "linedata and offsets must be 1D arrays");
    return nullptr;
  }
  if (PyArray_DIM(offsets, 0) < ny) {
    Py_DECREF(offsets);
    Py_DECREF(linedata);
    PyErr_SetString(PyExc_ValueError, "offsets array is shorter than ny");
    return nullptr;
  }
  if (ny < 1 || nx < 1) {
    Py_DECREF(offsets);
    Py_DECREF(linedata);
    PyErr_SetString(PyExc_ValueError, "ny and nx must be positive");
    return nullptr;
  }

  npy_intp dims[2] = {ny, nx};
  PyArrayObject *image = reinterpret_cast<PyArrayObject *>(
      PyArray_SimpleNew(2, dims, NPY_INT32));
  if (image == nullptr) {
    Py_DECREF(offsets);
    Py_DECREF(linedata);
    return nullptr;
  }

  rod_TY6_decompress(
      reinterpret_cast<std::int32_t *>(PyArray_DATA(image)),
      reinterpret_cast<const char *>(PyArray_DATA(linedata)),
      reinterpret_cast<const char *>(PyArray_DATA(offsets)),
      ny,
      nx);

  Py_DECREF(offsets);
  Py_DECREF(linedata);
  return reinterpret_cast<PyObject *>(image);
}

PyMethodDef methods[] = {
    {"decode_ty6_image", decode_ty6_image, METH_VARARGS, "Decode a TY6-compressed image."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "ty6_cpp",
    "Standalone C++ TY6 decompression backend.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit_ty6_cpp(void) {
  import_array();
  return PyModule_Create(&module);
}

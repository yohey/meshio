"""
I/O PLOT3D mesh format
<https://www.grc.nasa.gov/www/wind/valid/plot3d.html>
<https://vtk.org/doc/nightly/html/classvtkMultiBlockPLOT3DReader.html>
"""
import numpy as np

from .._helpers import register
from .._mesh import CellBlock, Mesh



class Info:
    """Info Container for the PLOT3D reader."""

    def __init__(self):
        self.double_precision = False
        self.formatted = True
        self.byte_order = "="
        self.has_byte_count = True
        self.i_blanking = False
        self.multi_block = True
        self.dimension = 3
        self.merge_blocks = True

        self.idtype = np.dtype(self.byte_order + "i4")
        self.fdtype = np.dtype(self.byte_order + ("f8" if self.double_precision else "f4"))



def read(filename):
    with open(filename, "rb") as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    info = Info()

    b = f.read()
    s = slice(0, 0)

    if info.multi_block:
        if info.has_byte_count:
            s = slice(s.stop, info.idtype.itemsize)
            n = np.frombuffer(b[s], info.idtype)[0]
            assert(n == info.idtype.itemsize)
            s = slice(s.stop, s.stop + n)
            num_blocks = np.frombuffer(b[s], info.idtype)[0]
            s = slice(s.stop, s.stop + info.idtype.itemsize)
            n = np.frombuffer(b[s], info.idtype)[0]
            assert(n == info.idtype.itemsize)
        else:
            s = slice(s.stop, info.idtype.itemsize)
            num_blocks = np.frombuffer(b[s], info.idtype)[0]
    else:
        num_blocks = 1

    if info.has_byte_count:
        s = slice(s.stop, s.stop + info.idtype.itemsize)
        n = np.frombuffer(b[s], info.idtype)[0]
        assert(n == info.idtype.itemsize * num_blocks * info.dimension)
        s = slice(s.stop, s.stop + n)
        nijk = np.frombuffer(b[s], info.idtype).reshape(num_blocks, info.dimension)
        s = slice(s.stop, s.stop + info.idtype.itemsize)
        n = np.frombuffer(b[s], info.idtype)[0]
        assert(n == info.idtype.itemsize * num_blocks * info.dimension)
    else:
        s = slice(s.stop, s.stop + info.idtype.itemsize * num_blocks * info.dimension)
        nijk = np.frombuffer(b[s], info.idtype).reshape(num_blocks, info.dimension)

    points = []
    cells = []
    cell_id = 0
    boundaries = []
    cell_data = []
    boundary_data = []
    boundary_data_id = num_blocks
    for ib in range(num_blocks):
        if info.has_byte_count:
            s = slice(s.stop, s.stop + info.idtype.itemsize)
            n = np.frombuffer(b[s], info.idtype)[0]
            assert(n == info.fdtype.itemsize * np.prod(nijk[ib]) * info.dimension)

            s = slice(s.stop, s.stop + n)
            xyz = np.frombuffer(b[s], info.fdtype).reshape(info.dimension, *reversed(nijk[ib]))

            s = slice(s.stop, s.stop + info.idtype.itemsize)
            n = np.frombuffer(b[s], info.idtype)[0]
            assert(n == info.fdtype.itemsize * np.prod(nijk[ib]) * info.dimension)

        _points = xyz.transpose(1, 2, 3, 0).reshape(np.prod(nijk[ib]), info.dimension)
        points.extend(_points)

        _cell_type, _cells = _generate_cells(nijk[ib])
        _cells += cell_id
        cell_id = _cells.max() + 1
        cells.append([_cell_type, _cells.reshape(np.prod(_cells.shape[:-1]), _cells.shape[-1])])
        _cell_data = np.full(np.prod(_cells.shape[:-1]), ib, dtype=int)
        cell_data.append(_cell_data)

        _boundary_type, _boundaries, _boundary_data = _generate_boundaries(_cells)
        boundaries.append([_boundary_type, _boundaries])
        _boundary_data += boundary_data_id
        boundary_data_id = _boundary_data.max() + 1
        boundary_data.append(_boundary_data)

    if info.merge_blocks:
        points, cells, boundaries = _remove_duplicates(points, cells, boundaries)

    return Mesh(points, cells + boundaries, cell_data={"plot3d:tag": cell_data + boundary_data})


def _check_formatted(f):
    return True


def _check_byte_order(f):
    return True


def _check_byte_count(f):
    return True


def _check_blanking_and_precision(f):
    return True


def _check_multi_grid(f):
    return True


def _check_dimension(f):
    return True


def _generate_cells(nijk):
    ni = nijk[0] - 1
    nj = nijk[1] - 1
    i = np.arange(ni).reshape(1, ni)
    j = np.arange(nj).reshape(nj, 1)

    if sum(nijk > 1) == 2: # 2D
        cells = np.empty([nj, ni, 4], dtype=int)
        cells[:, :, 0] = i + (ni + 1) * j
        cells[:, :, 1] = cells[:, :, 0] + 1
        cells[:, :, 2] = cells[:, :, 1] + ni + 1
        cells[:, :, 3] = cells[:, :, 2] - 1
        cell_type = "quad"

    else: # 3D
        nk = nijk[2] - 1
        k = np.arange(nk)
        cells = np.zeros([nk, nj, ni, 8], dtype=int)
        raise NotImplementedError

    return cell_type, cells


def _generate_boundaries(cells):
    boundaries = []
    boundary_data = []

    if len(cells.shape) == 3: # 2D
        boundary_type = "line"
        nj, ni, _ = cells.shape

        lines = np.empty([nj, 2], dtype=int)
        lines[:, 0] = cells[:, 0, 0]
        lines[:, 1] = cells[:, 0, 3]
        boundaries.append(lines)
        boundary_data.append(np.full(nj, 0, dtype=int))

        lines = np.empty([nj, 2], dtype=int)
        lines[:, 0] = cells[:, -1, 1]
        lines[:, 1] = cells[:, -1, 2]
        boundaries.append(lines)
        boundary_data.append(np.full(nj, 1, dtype=int))

        lines = np.empty([ni, 2], dtype=int)
        lines[:, 0] = cells[0, :, 0]
        lines[:, 1] = cells[0, :, 1]
        boundaries.append(lines)
        boundary_data.append(np.full(ni, 2, dtype=int))

        lines = np.empty([ni, 2], dtype=int)
        lines[:, 0] = cells[-1, :, 3]
        lines[:, 1] = cells[-1, :, 2]
        boundaries.append(lines)
        boundary_data.append(np.full(ni, 3, dtype=int))

    else: # 3D
        boundary_type = "quad"
        nk, nj, ni, _ = cells.shape
        raise NotImplementedError

    return boundary_type, np.concatenate(boundaries), np.concatenate(boundary_data)


def _remove_duplicates(points, cells, boundaries):
    points = np.array(points)
    points, indices, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    indices = indices.argsort()
    points = points[indices]

    inverse = indices.argsort()[inverse]

    for _cell_type, _cells in cells:
        _cells[:, :] = inverse[_cells.ravel()].reshape(_cells.shape)

    for _boundary_type, _boundaries in boundaries:
        _boundaries[:, :] = inverse[_boundaries.ravel()].reshape(_boundaries.shape)

    return points, cells, boundaries



def write(filename, mesh, double_precision=True, formatted=True, byte_order="BigEndian", has_byte_count=False, i_blanking=False):

    # with open(filename, "wb") as f:
    #     pass

    raise NotImplementedError



register("plot3d", [".xyz"], read, {"plo3d": write})

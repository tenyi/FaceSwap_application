'''
Author: YadiraF 
Mail: fengyao@sjtu.edu.cn
'''
import numpy as np

def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: [u, v] or [x, y] 
        tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[:,2] - tp[:,0]
    v1 = tp[:,1] - tp[:,0]
    v2 = point - tp[:,0]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)

def get_point_weight(point, tri_points):
    ''' Get the weights of the position
    Methods: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
     -m1.compute the area of the triangles formed by embedding the point P inside the triangle
     -m2.Christer Ericson's book "Real-Time Collision Detection". faster, so I used this.
    Args:
        point: [u, v] or [x, y] 
        tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
    Returns:
        w0: weight of v0
        w1: weight of v1
        w2: weight of v3
     '''
    tp = tri_points
    # vectors
    v0 = tp[:,2] - tp[:,0]
    v1 = tp[:,1] - tp[:,0]
    v2 = point - tp[:,0]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2


def _barycentric_weights(p0, p1, p2, u, v):
    '''Barycentric weights of the pixel inside the triangle.

    Args:
        p0, p1, p2: (N, 2) arrays with the triangle vertices.
        u, v: (N,) arrays with the pixel coordinates.

    Returns:
        (w0, w1, w2): (N,) arrays, the weight of every vertex.
    '''
    a = p2 - p0
    b = p1 - p0
    c = np.stack([u - p0[:, 0], v - p0[:, 1]], axis=1)

    dot_aa = a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1]
    dot_ab = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    dot_ac = a[:, 0] * c[:, 0] + a[:, 1] * c[:, 1]
    dot_bb = b[:, 0] * b[:, 0] + b[:, 1] * b[:, 1]
    dot_bc = b[:, 0] * c[:, 0] + b[:, 1] * c[:, 1]

    denom = dot_aa * dot_bb - dot_ab * dot_ab
    # Degenerate triangles keep the original behaviour (inverDeno = 0).
    inv = np.zeros_like(denom)
    np.divide(1.0, denom, out=inv, where=(denom != 0))

    w1 = (dot_bb * dot_ac - dot_ab * dot_bc) * inv
    w2 = (dot_aa * dot_bc - dot_ab * dot_ac) * inv
    return 1.0 - w1 - w2, w1, w2


def render_texture(vertices, colors, triangles, h, w, c = 3, block = 20000,
                   shading = 'gouraud'):
    ''' render mesh by z buffer
    Args:
        vertices: 3 x nver
        colors: 3 x nver
        triangles: 3 x ntri
        h: height
        w: width
        shading: 'gouraud' interpolates the vertex colours inside every
            triangle, 'flat' uses the average colour of the three vertices
            (the behaviour of the original implementation).

    This is a vectorised rewrite of the original per-triangle / per-pixel
    Python loops.  The z-buffer semantics are kept identical: for every pixel
    the triangle with the greatest depth wins, and ties are resolved in favour
    of the lowest triangle index.
    '''
    # initial
    image = np.zeros((h, w, c))
    depth_buffer = np.zeros([h, w]) - 999999.

    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3.
    tri_tex = (colors[:, triangles[0,:]] + colors[:,triangles[1,:]] + colors[:, triangles[2,:]])/3.

    ntri = triangles.shape[1]
    if ntri == 0:
        return image

    # inner bounding box of every triangle (same clamping as the original loop:
    # the lower bound is only clamped from below, the upper bound only from above)
    vx = vertices[0, triangles]
    vy = vertices[1, triangles]
    umin = np.maximum(np.ceil(vx.min(axis=0)).astype(np.int64), 0)
    umax = np.minimum(np.floor(vx.max(axis=0)).astype(np.int64), w - 1)
    vmin = np.maximum(np.ceil(vy.min(axis=0)).astype(np.int64), 0)
    vmax = np.minimum(np.floor(vy.max(axis=0)).astype(np.int64), h - 1)

    p0 = vertices[:2, triangles[0]].T.astype(np.float64)
    p1 = vertices[:2, triangles[1]].T.astype(np.float64)
    p2 = vertices[:2, triangles[2]].T.astype(np.float64)

    c0 = colors[:, triangles[0]].T.astype(np.float64)
    c1 = colors[:, triangles[1]].T.astype(np.float64)
    c2 = colors[:, triangles[2]].T.astype(np.float64)

    for start in range(0, ntri, block):
        stop = min(start + block, ntri)
        sl = slice(start, stop)

        bw = np.maximum(umax[sl] - umin[sl] + 1, 0)
        bh = np.maximum(vmax[sl] - vmin[sl] + 1, 0)
        counts = bw * bh
        total = int(counts.sum())
        if total == 0:
            continue

        tri_idx = np.repeat(np.arange(start, stop), counts)
        offset = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        bw_rep = np.repeat(bw, counts)
        u = np.repeat(umin[sl], counts) + offset % bw_rep
        v = np.repeat(vmin[sl], counts) + offset // bw_rep

        w0, w1, w2 = _barycentric_weights(p0[tri_idx], p1[tri_idx], p2[tri_idx],
                                          u.astype(np.float64), v.astype(np.float64))
        inside = (w1 >= 0) & (w2 >= 0) & (w1 + w2 < 1)
        if not inside.any():
            continue

        # keep only the pixels that really fall inside their triangle
        keep = np.nonzero(inside)[0]
        tri_idx = tri_idx[keep]
        u = u[keep]
        v = v[keep]
        w0 = w0[keep]
        w1 = w1[keep]
        w2 = w2[keep]

        pix = v * w + u
        depth = tri_depth[tri_idx]

        # keep the deepest triangle for every pixel of this block
        order = np.lexsort((-depth, pix))
        pix_s = pix[order]
        first = np.ones(pix_s.shape, dtype=bool)
        first[1:] = pix_s[1:] != pix_s[:-1]
        sel = order[first]

        px = u[sel]
        py = v[sel]
        pdepth = depth[sel]
        better = pdepth > depth_buffer[py, px]
        if not better.any():
            continue
        px = px[better]
        py = py[better]
        pdepth = pdepth[better]
        ptri = tri_idx[sel][better]

        depth_buffer[py, px] = pdepth
        if shading == 'gouraud':
            sw0 = w0[sel][better][:, np.newaxis]
            sw1 = w1[sel][better][:, np.newaxis]
            sw2 = w2[sel][better][:, np.newaxis]
            image[py, px, :] = (sw0 * c0[ptri] + sw1 * c1[ptri] + sw2 * c2[ptri])
        else:
            image[py, px, :] = tri_tex[:, ptri].T

    return image


def map_texture(src_image, src_vertices, dst_vertices, dst_triangle_buffer, triangles, h, w, c = 3, mapping_type = 'bilinear'):
    '''
    Args:
        triangles: 3 x ntri

        # src
        src_image: height x width x nchannels
        src_vertices: 3 x nver
        
        # dst
        dst_vertices: 3 x nver
        dst_triangle_buffer: height x width. the triangle index of each pixel in dst image

    Returns:
        dst_image: height x width x nchannels

    '''
    [sh, sw, sc] = src_image.shape
    dst_image = np.zeros((h, w, c))
    for y in range(h):
        for x in range(w):
            tri_ind = dst_triangle_buffer[y,x]
            if tri_ind < 0: # no tri in dst image
                continue 
            #if src_triangles_vis[tri_ind]: # the corresponding triangle in src image is invisible
            #   continue
            
            # then. For this triangle index, map corresponding pixels(in triangles) in src image to dst image
            # Two Methods:
            # M1. Calculate the corresponding affine matrix from src triangle to dst triangle. Then find the corresponding src position of this dst pixel.
            # -- ToDo
            # M2. Calculate the relative position of three vertices in dst triangle, then find the corresponding src position relative to three src vertices.
            tri = triangles[:, tri_ind]
            # dst weight, here directly use the center to approximate because the tri is small
            # if tri_ind < 366:
                # print tri_ind
            w0, w1, w2 = get_point_weight([x, y], dst_vertices[:2, tri])
            # else:
            #     w0 = w1 = w2 = 1./3
            # src
            src_texel = w0*src_vertices[:2, tri[0]] + w1*src_vertices[:2, tri[1]] + w2*src_vertices[:2, tri[2]] #
# 
            if src_texel[0] < 0 or src_texel[0]> sw-1 or src_texel[1]<0 or src_texel[1] > sh-1:
                dst_image[y, x, :] = 0
                continue
            # As the coordinates of the transformed pixel in the image will most likely not lie on a texel, we have to choose how to
            # calculate the pixel colors depending on the next texels
            # there are three different texture interpolation methods: area, bilinear and nearest neighbour
            # print y, x, src_texel
            # nearest neighbour 
            if mapping_type == 'nearest':
                dst_image[y, x, :] = src_image[int(round(src_texel[1])), int(round(src_texel[0])), :]
            # bilinear
            elif mapping_type == 'bilinear':
                # next 4 pixels
                ul = src_image[int(np.floor(src_texel[1])), int(np.floor(src_texel[0])), :]
                ur = src_image[int(np.floor(src_texel[1])), int(np.ceil(src_texel[0])), :]
                dl = src_image[int(np.ceil(src_texel[1])), int(np.floor(src_texel[0])), :]
                dr = src_image[int(np.ceil(src_texel[1])), int(np.ceil(src_texel[0])), :]

                yd = src_texel[1] - np.floor(src_texel[1])
                xd = src_texel[0] - np.floor(src_texel[0])
                dst_image[y, x, :] = ul*(1-xd)*(1-yd) + ur*xd*(1-yd) + dl*(1-xd)*yd + dr*xd*yd
                
    return dst_image


def get_depth_buffer(vertices, triangles, h, w):
    '''
    Args:
        vertices: 3 x nver
        triangles: 3 x ntri
        h: height
        w: width
    Returns:
        depth_buffer: height x width
    ToDo:
        whether to add x, y by 0.5? the center of the pixel?
        m3. like somewhere is wrong
    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    # Here, the bigger the z, the fronter the point.
    '''
    # initial 
    depth_buffer = np.zeros([h, w]) - 999999. #+ np.min(vertices[2,:]) - 999999. # set the initial z to the farest position

    ## calculate the depth(z) of each triangle
    #-m1. z = the center of shpere(through 3 vertices)
    #center3d = (vertices[:, triangles[0,:]] + vertices[:,triangles[1,:]] + vertices[:, triangles[2,:]])/3.
    #tri_depth = np.sum(center3d**2, axis = 0)
    #-m2. z = the center of z(v0, v1, v2)
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3.
    
    for i in range(triangles.shape[1]):
        tri = triangles[:, i] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[0,tri]))), 0)
        umax = min(int(np.floor(np.max(vertices[0,tri]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[1,tri]))), 0)
        vmax = min(int(np.floor(np.max(vertices[1,tri]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                #-m3. calculate the accurate depth(z) of each pixel by barycentric weights
                #w0, w1, w2 = weightsOfpoint([u,v], vertices[:2, tri])
                #tri_depth = w0*vertices[2,tri[0]] + w1*vertices[2,tri[1]] + w2*vertices[2,tri[2]]
                if tri_depth[i] > depth_buffer[v, u]: # and is_pointIntri([u,v], vertices[:2, tri]): 
                    depth_buffer[v, u] = tri_depth[i]

    return depth_buffer


def get_triangle_buffer(vertices, triangles, h, w):
    '''
    Args:
        vertices: 3 x nver
        triangles: 3 x ntri
        h: height
        w: width
    Returns:
        depth_buffer: height x width
    ToDo:
        whether to add x, y by 0.5? the center of the pixel?
        m3. like somewhere is wrong
    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    # Here, the bigger the z, the fronter the point.
    '''
    # initial 
    depth_buffer = np.zeros([h, w]) - 999999. #+ np.min(vertices[2,:]) - 999999. # set the initial z to the farest position
    triangle_buffer = np.zeros_like(depth_buffer, dtype = np.int32) - 1 # if -1, the pixel has no triangle correspondance

    ## calculate the depth(z) of each triangle
    #-m1. z = the center of shpere(through 3 vertices)
    #center3d = (vertices[:, triangles[0,:]] + vertices[:,triangles[1,:]] + vertices[:, triangles[2,:]])/3.
    #tri_depth = np.sum(center3d**2, axis = 0)
    #-m2. z = the center of z(v0, v1, v2)
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3.
    
    for i in range(triangles.shape[1]):
        tri = triangles[:, i] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[0,tri]))), 0)
        umax = min(int(np.floor(np.max(vertices[0,tri]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[1,tri]))), 0)
        vmax = min(int(np.floor(np.max(vertices[1,tri]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                #-m3. calculate the accurate depth(z) of each pixel by barycentric weights
                #w0, w1, w2 = weightsOfpoint([u,v], vertices[:2, tri])
                #tri_depth = w0*vertices[2,tri[0]] + w1*vertices[2,tri[1]] + w2*vertices[2,tri[2]]
                if tri_depth[i] > depth_buffer[v, u] and isPointInTri([u,v], vertices[:2, tri]): 
                    depth_buffer[v, u] = tri_depth[i]
                    triangle_buffer[v, u] = i

    return triangle_buffer


def vis_of_vertices(vertices, triangles, h, w, depth_buffer = None):
    '''
    Args:
        vertices: 3 x nver
        triangles: 3 x ntri
        depth_buffer: height x width
    Returns:
        vertices_vis: nver. the visibility of each vertex
    '''
    if depth_buffer == None:
        depth_buffer = get_depth_buffer(vertices, triangles, h, w)

    vertices_vis = np.zeros(vertices.shape[1], dtype = bool)
    
    depth_tmp = np.zeros_like(depth_buffer) - 99999
    for i in range(vertices.shape[1]):
        vertex = vertices[:, i]

        if np.floor(vertex[0]) < 0 or np.ceil(vertex[0]) > w-1 or np.floor(vertex[1]) < 0 or np.ceil(vertex[1]) > h-1:
            continue        
        
        # bilinear interp 
        # ul = depth_buffer[int(np.floor(vertex[1])), int(np.floor(vertex[0]))]
        # ur = depth_buffer[int(np.floor(vertex[1])), int(np.ceil(vertex[0]))]
        # dl = depth_buffer[int(np.ceil(vertex[1])), int(np.floor(vertex[0]))]
        # dr = depth_buffer[int(np.ceil(vertex[1])), int(np.ceil(vertex[0]))]
        
        # yd = vertex[1] - np.floor(vertex[1])
        # xd = vertex[0] - np.floor(vertex[0])

        # vertex_depth = ul*(1-xd)*(1-yd) + ur*xd*(1-yd) + dl*(1-xd)*yd + dr*xd*yd
        
        # nearest
        px = int(np.round(vertex[0]))
        py = int(np.round(vertex[1]))

        # if (vertex[2] > depth_buffer[ul[0], ul[1]]) & (vertex[2] > depth_buffer[ur[0], ur[1]]) & (vertex[2] > depth_buffer[dl[0], dl[1]]) & (vertex[2] > depth_buffer[dr[0], dr[1]]):
        if vertex[2] < depth_tmp[py, px]:
            continue
        
        # if vertex[2] > depth_buffer[py, px]:
        #     vertices_vis[i] = True
        #     depth_tmp[py, px] = vertex[2]
        # elif np.abs(vertex[2] - depth_buffer[py, px]) < 1:
        #     vertices_vis[i] = True

        threshold = 2 # need to be optimized.
        if np.abs(vertex[2] - depth_buffer[py, px]) < threshold:
        # if np.abs(vertex[2] - vertex_depth) < threshold:
            vertices_vis[i] = True
            depth_tmp[py, px] = vertex[2]

    return vertices_vis

# Written by: Nick Gerend, @dataoutsider
# Viz: "Up, Up and Away", enjoy!

import pandas as pd
import numpy as np
import os
from datetime import datetime
import scipy.optimize as optimize
from math import pi, cos, sin, sqrt, exp, log
import matplotlib.pyplot as plt

class point:
    def __init__(self, index, group, x, y, path, z=0., row=0, column=0, item='', group_id=0): 
        self.index = index
        self.group = group
        self.x = x
        self.y = y
        self.path = path
        self.z = z
        self.row = row
        self.column = column
        self.item = item
        self.group_id = group_id
    def to_dict(self):
        return {
            'index' : self.index,
            'group' : self.group,
            'x' : self.x,
            'y' : self.y,
            'path' : self.path,
            'z' : self.z,
            'row' : self.row,
            'column' : self.column,
            'item' : self.item,
            'group_id' : self.group_id}

#region functions
def rescale(x, xmin, xmax, newmin, newmax):
    rescaled = (newmax-newmin)*((x-xmin)/(xmax-xmin))+newmin
    return rescaled

#region profile
def teardrop_yz(y, miny, maxy, minz, maxz, m, f=1.):
    zs = rescale(y, miny, maxy, -1., 1.) # original y as z
    a1 = 0.
    a2 = 180.
    a = teardrop_angle(a1, a2, zs)
    theta = a*pi/180.
    z = -cos(theta)
    zs =  rescale(z, -1., 1., minz, maxz)
    yt = np.sin(theta)*np.sin(0.5*theta)**m*f
    yts = rescale(yt, 0., 1.0, miny, maxy)
    return yts, zs, a

def angle_func(a, y):
    at = -np.cos(a*np.pi/180,)
    at -= y
    return at

def teardrop_angle(a1, a2, y):
    a = optimize.bisect(angle_func, a1, a2, args=(y), maxiter=1000)
    return a

def profile(list_xy, miny, maxy, minz, maxz):
    alast = 0.
    for i in range(len(list_xy)):
        y = list_xy[i].y
        if i==20:
            stop = 0
        yt, z, alast = teardrop_yz(y, miny, maxy, minz, maxz, 1.5, f=1.2) # balloon shape: y, 0., 10., 0., 8., 1.5, f=1.2
        list_xy[i].y = yt
        list_xy[i].z = z
    return list_xy

#extras:

def balloon(a, b, points, t1_deg=0., t2_deg=360.):
    t1 = t1_deg*pi/180.
    t2 = t2_deg*pi/180.
    theta = np.linspace(t1, t2, num=points)
    r = 1 + a*(1/np.cosh(b*(theta-pi)))
    y = r*np.cos(theta)
    x = r*np.sin(theta)
    return list(zip(x,y))

def teardrop(m, points, t1_deg=0., t2_deg=360., f=1.):
    t1 = t1_deg*pi/180.
    t2 = t2_deg*pi/180.
    theta = np.linspace(t1, t2, num=points)
    y = -np.cos(theta)
    x = np.sin(theta)*np.sin(0.5*theta)**m*f
    return list(zip(x,y))
#endregion

#region grid patterns
def grid_stripes(columns, r1, r2, width, points=10):
    list_xy = []
    ix = 0
    start_x = 0.
    path = 1
    width /= columns
    for i in range(columns):
        xs = np.linspace(start_x, start_x+width, num=points)
        xsr = np.linspace(start_x+width, start_x, num=points)
        ys = np.linspace(r1, r2, num=points)
        ysr = np.linspace(r2, r1, num=points)
        path = 1
        for j in range(points-1):
            list_xy.append(point(ix, i, xs[j], r1, path, 0., 0, i))
            path +=1
            ix += 1
        for j in range(points-1):
            list_xy.append(point(ix, i, start_x+width, ys[j], path, 0., 0, i))
            path +=1
            ix += 1
        for j in range(points-1):
            list_xy.append(point(ix, i, xsr[j], r2, path, 0., 0, i))
            path +=1
            ix += 1
        for j in range(points-1):
            list_xy.append(point(ix, i, start_x, ysr[j], path, 0., 0, i))
            path +=1
            ix += 1
        start_x += width        
    return list_xy

def grid_squares(rows, columns, r1, r2, width, points=10):
    list_xy = []
    ix = 0
    path = 1
    height = (r2-r1)/rows
    width /= columns
    for k in range(rows):
        start_x = 0.
        for i in range(columns):
            xs = np.linspace(start_x, start_x+width, num=points)
            xsr = np.linspace(start_x+width, start_x, num=points)
            ys = np.linspace(r1, r1+height, num=points)
            ysr = np.linspace(r1+height, r1, num=points)
            path = 1
            for j in range(points-1):
                list_xy.append(point(ix, k*columns+i, xs[j], r1, path, 0., k, i))
                path +=1
                ix += 1
            for j in range(points-1):
                list_xy.append(point(ix, k*columns+i, start_x+width, ys[j], path, 0., k, i))
                path +=1
                ix += 1
            for j in range(points-1):
                list_xy.append(point(ix, k*columns+i, xsr[j], r1+height, path, 0., k, i))
                path +=1
                ix += 1
            for j in range(points-1):
                list_xy.append(point(ix, k*columns+i, start_x, ysr[j], path, 0., k, i))
                path +=1
                ix += 1
            start_x += width
        r1 += height      
    return list_xy

def points_between_two_points(x1, y1, x2, y2, points):
    return list(zip(np.linspace(x1, x2, num=points), np.linspace(y1, y2, num=points)))

def grid_diamonds(rows, columns, r1, r2, width, points=10):
    # good ratio: h = 5. to w = 8.
    h = (r2-r1)/(rows) #(r2-r1)/(rows)    (r2-r1)/(rows+1)
    w = width/columns
    #dipoly = [[0.,0.],[w/2.,h],[w,0.0],[w/2.,-h],[0.,0.]]
    dipoly = [[-w/2,h],[0.,2.*h],[w/2.,h],[0.,0.],[-w/2,h]]
    #dipoly = [[-w/2,0.],[0.,h],[w/2.,0.],[0.,-h],[-w/2,0.]]
    UL = points_between_two_points(dipoly[0][0], dipoly[0][1], dipoly[1][0], dipoly[1][1], points)
    UR = points_between_two_points(dipoly[1][0], dipoly[1][1], dipoly[2][0], dipoly[2][1], points)
    LR = points_between_two_points(dipoly[2][0], dipoly[2][1], dipoly[3][0], dipoly[3][1], points)
    LL = points_between_two_points(dipoly[3][0], dipoly[3][1], dipoly[4][0], dipoly[4][1], points)
    list_dipoly = []
    for i in range(len(UL)):
        list_dipoly.append(UL[i])
    for i in range(len(UR)-1):
        list_dipoly.append(UR[i+1])
    for i in range(len(LR)-1):
        list_dipoly.append(LR[i+1])
    for i in range(len(LL)):
        list_dipoly.append(LL[i])
    ix = 0
    item = 1
    list_xy = []
    x_adj = 0
    y_adj = 0
    offset = 0
    for i in range(rows):
        for j in range(columns):
            for k in range(len(list_dipoly)):
                list_xy.append(point(ix, item, list_dipoly[k][0]+x_adj+offset, list_dipoly[k][1]+y_adj+r1, k, 0., i, j))
                ix += 1
            item += 1
            x_adj += w
        x_adj = 0
        if offset == 0:
            offset = w/2.
        else:
            offset = 0
        y_adj += h
    list_xy = [i for i in list_xy if (i.y >= 0.) and (i.y <= r2+0.001)]
    return list_xy

def fan(points):
    if points % 2 != 0:
        points += 1
    x = []
    y = []
    path = []
    angle = -90.
    path_i = 1
    for i in range(points+1):
        x.append(sin(angle*pi/180.))
        y.append(cos(angle*pi/180.))
        path.append(path_i)
        angle += 1./points*180.
        path_i += 1
    path_i = int(points*2-points/2)+1
    for i in range(int(points/2)):
        x.append(x[i]+1.)  
        y.append(y[i]-1.)
        path.append(path_i)
        path_i -= 1
    path_i = int(points*2-1)+1
    for i in range(int(points/2)-1):
            x.append(x[i+(int(points/2)+1)]-1.)
            y.append(y[i+(int(points/2)+1)]-1.)
            path.append(path_i)
            path_i -= 1
    x.append(x[0])  
    y.append(y[0])
    path.append(points*2+1)
    return x,y,path

def grid_fans(rows, columns, r1, r2, width, points=40):
    x,y,p = fan(points)
    h = (r2-r1)/((rows)/2.) #(r2-r1)/((rows+1)/2.)
    w = width/columns
    x = [(i/2.)*w for i in x]
    y = [(i/2.+0.5)*h for i in y] #[(i/2.+0.5)*h for i in y]
    list_xy = []
    ix = 0
    item = 1
    xt = 0
    yt = 0
    offset = 0
    for i in range(rows):
        for j in range(columns):      
            for k in range(len(p)):
                list_xy.append(point(ix, item, x[k]+xt+offset, y[k]+yt, p[k], 0., i, j))
                ix +=1
            item += 1
            xt += w
        xt = 0
        if offset == 0:
            offset = w/2.
        else:
            offset = 0
        yt += h/2.
    list_xy = [i for i in list_xy if (i.y >= 0.) and (i.y <= r2+0.001)]
    return list_xy

def sigmoid_xy(x1, y1, x2, y2, points, orientation = 'h', limit = 6):
    x_1 = x1
    y_1 = y1
    x_2 = x2
    y_2 = y2
    if orientation == 'v':
        x1 = y_1
        y1 = x_1
        x2 = y_2
        y2 = x_2
    x = []
    y = []
    amin = 1./(1.+exp(limit))
    amax = 1./(1.+exp(-limit))
    da = amax-amin
    for i in range(points):
        i += 1
        xi = (i-1.)*((2.*limit)/(points-1.))-limit
        yi = ((1.0/(1.0+exp(-xi)))-amin)/da
        x.append((xi-(-limit))/(2.*limit)*(x2-x1)+x1)
        y.append((yi-(0.))/(1.)*(y2-y1)+y1)
    return { 'h': list(zip(x,y)), 'v': list(zip(y,x))}.get(orientation, None)

def ornament_xy(width, height, quad_points, limit = 4.):
    SW = sigmoid_xy(0., 0., -width/2., height/2., quad_points, 'v', limit)
    NW = sigmoid_xy(-width/2., height/2, 0., height, quad_points, 'v', limit)  
    NE = sigmoid_xy(0., height,width/2, height/2,  quad_points, 'v', limit)
    SE = sigmoid_xy(width/2, height/2, 0., 0., quad_points, 'v', limit)
    list_xy = SW + NW + NE + SE
    return list_xy

def grid_ornaments(rows, columns, r1, r2, width, points=40):
    r = rows
    c = columns
    height = (r2-r1)/(rows)*2 #height = (r2-r1)/(rows+1)*2
    width = width/(columns)
    list_o_xy = ornament_xy(width, height, 10, 4.)
    x_shift = 0.
    y_shift = r1 #-height/2.
    list_xy = []
    ix = 0
    item = 1
    for i in range(rows): #+2
        for j in range(columns): #+1
            for k in range(len(list_o_xy)):
                list_xy.append(point(ix, item, list_o_xy[k][0]+x_shift, list_o_xy[k][1]+y_shift, k, 0., i, j))
            x_shift += width
            item += 1
        if i % 2 == 0:
            x_shift = 0.5*width
            #rows -= 1
        else:
            x_shift = 0.
            #rows += 1
        y_shift += height/2
    #list_xy_lattice = [i for i in list_xy if (i.x >= 0.) and (i.x <= r)and (i.y >= 0.) and (i.y <= c)]
    list_xy_lattice = [i for i in list_xy]
    list_xy_lattice = [i for i in list_xy_lattice if (i.y >= 0.) and (i.y <= r2+0.001)]
    return list_xy_lattice
#endregion

def radial_xy(list_xy, N, offset=0., min_x = 0.):
    for i in range(len(list_xy)):
        x = list_xy[i].x
        y = list_xy[i].y
        angle = (2.*pi)*(((x-min_x)%(N))/(N))
        angle_deg = angle * 180./pi
        angle_rotated = (abs(angle_deg-360.)+90.) % 360. 
        angle_new = angle_rotated * pi/180.
        x_out = (offset+y)*cos(angle_new)
        y_out = (offset+y)*sin(angle_new)
        list_xy[i].x = x_out
        list_xy[i].y = y_out
    return list_xy
#endregion

#region test
# columns = 35 # 19 for stripes
# rows = 20 # 20
# r1 = 0. # 0
# r2 = 10. # 10
# w = 8. # 1
# points = 20 # 500 for stripes

# columns = 8 # 19 for stripes
# rows = 6 # 20
# r1 = 0. # 0
# r2 = 10. # 10
# w = 8. # 1
# points = 20 # 500 for stripes

# #region transform y, z
# #list_xy = grid_stripes(columns, r1, r2, w, points)
# #list_xy = grid_squares(rows, columns, r1, r2, w, points)
# #list_xy = grid_diamonds(rows, columns, r1, r2, w, points)
# list_xy = grid_fans(rows, columns, r1, r2, w, points)
# #list_xy = grid_ornaments(rows, columns, r1, r2, w, points)

# x = [o.x for o in list_xy]
# y = [o.y for o in list_xy]
# plt.scatter(x, y)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

# list_xy = profile(list_xy, 0., r2, 0., r2)
# list_xy = radial_xy(list_xy, w)
#endregion
#endregion

# df_out = pd.DataFrame.from_records([s.to_dict() for s in list_xy])
# df_out.to_csv(os.path.dirname(__file__) + '/teardrop.csv', encoding='utf-8', index=False)

df_b = pd.read_csv(os.path.dirname(__file__) + '/aircraft_balloons.csv')
df_s = pd.read_csv(os.path.dirname(__file__) + '/states_mod.csv')
df = pd.merge(df_b, df_s, left_on=['STATE'], right_on = ['State'], how='inner')
#df = df.loc[df['State']=='SD']
df_group = df.groupby(['Longitude', 'State'], sort=True)

#region algorithm
list_xyz = []
for i, row in df_s.iterrows():
    state = row['State']
    lat = row['Latitude']
    long = row['Longitude']
    count = row['Count']
    list_xyz.append(point(count, -1, 0., lat, -1, long, -1, -1, state))
y_i = 0.
z_i = 0.
y_last = 0.
col_row_ratio = 1.75
last_lat = 0.
iter = 1
same_sign = False
sign_factor = 1.5
sign = 'up'
last_sign = 'down'
for index, items in df_group:
    size = len(items)
    rows = int(sqrt((size*2)/col_row_ratio))+1
    columns = int(rows*col_row_ratio)+1
    size_2 = ((log(size)+1)*10)**1.1*0.75
    state = items['State'].min()
    print(str(state) + ' ' + str(iter) + '.  size: ' + str(size) + '  rows: ' + str(rows) + '  columns: ' + str(columns) + '  half_r*c: ' + str(rows*columns/2.))
    state_list_xyz = []
    min_AWY = items['A_W_Date_Year'].min()  
    if size <= 25:
        state_list_xyz = grid_stripes(size*2, 0., size_2, 10., size+100)
    elif size <= 50:
        state_list_xyz = grid_diamonds(rows, columns, 0., size_2, 10., int(size/6.)) #6
    elif size <= 100:
        state_list_xyz = grid_fans(rows, columns, 0., size_2, 10., int(size/7.)) #7
    elif size <= 200:
        state_list_xyz = grid_ornaments(rows, columns, 0., size_2, 10., int(size/14.)) #8
    else:
        state_list_xyz = grid_squares(rows, columns, 0., size_2, 10., int(size/28.)) #9
    state_list_xyz = profile(state_list_xyz, 0., size_2+.002, 0., 2*size_2+.002) # add small fraction for floating points
    state_list_xyz = radial_xy(state_list_xyz, 10.)  
    lat = items['Latitude'].min()
    long = items['Longitude'].min()
    y_i = lat
    z_i = long
    group_id = 0
    last_group = ''
    for i in range(len(state_list_xyz)):
        state_list_xyz[i].y += y_i
        state_list_xyz[i].z += z_i
        state_list_xyz[i].item = state
        group = state_list_xyz[i].group
        x = state_list_xyz[i].x
        if i > 0 and x >= 0:
            if last_group != group:
                group_id += 1
        if i > 0 and x >= 0:
            state_list_xyz[i].group_id = group_id
        else:
            state_list_xyz[i].group_id = -1
        last_group = group
        list_xyz.append(state_list_xyz[i])
    # if iter == 8:
    #     break
    iter += 1

#endregion

df_out = pd.DataFrame.from_records([s.to_dict() for s in list_xyz])
df_out.to_csv(os.path.dirname(__file__) + '/teardrops.csv', encoding='utf-8', index=False)

print('finished')
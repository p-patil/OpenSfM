import sys, os
import numpy as np
import glob
import pickle
import cv2
import matplotlib.pyplot as plt
import json
import math
from mpl_toolkits.mplot3d import Axes3D
#import seaborn

def get_gps(json_path, video_filename):
    with open(json_path) as data_file:    
        data = json.load(data_file)
    
    seg = [x for x in data['segments'] if x['filename']==video_filename]
    assert(len(seg)==1)
    seg = seg[0]
    locs = seg['locations']
    loc2nparray = lambda locs, key: np.array([x[key] for x in locs]).ravel()
    
    res = {}
    bad_video_c = 0
    bad_video_t = 0
    bad_video_same = 0
    for ifile, f in enumerate(locs):
        if int(f['course']) == -1 or int(f['speed']) == -1:
            bad_video_c += 1   # Changed for interpolation
            if bad_video_c >= 3:
                break
        if ifile != 0:
            if int(f['timestamp'])-prev_t > 1100:
                bad_video_t = 1
                break
            if abs(int(f['timestamp']) - int(prev_t)) < 1:
                bad_video_same = 1
                break
        prev_t = f['timestamp']
    if bad_video_c >= 3:
        print('This is a bad video because course or speed is -1', json_path, video_filename)
        return None
    if bad_video_t:
        print('This is a bad video because time sample not uniform', json_path, video_filename)
        return None
    if len(locs)==0:
        print('This is a bad video because no location data available', json_path, video_filename)
        return None
    if bad_video_same:
        print('This is a bad video because same timestamps', json_path, video_filename)
        return None
    
    for key in locs[0].keys():
        res[key]=loc2nparray(locs, key)

    # add the starting time point and ending time point as well
    res['startTime'] = seg['startTime']
    res['endTime'] = seg['endTime']
    
    if res['timestamp'][0] - res['startTime'] > 2000:
        print('This is bad video because starting time too far ahead', json_path, video_filename)
        return None

    if res['endTime'] - res['timestamp'][-1] > 2000:
        print('This is bad video because ending time too far ahead', json_path, video_filename)
        return None

    return res

def fill_missing_speeds_and_courses(values, show_warning):
    l = len(values)
    for i in range(l):
        if values[i] == -1:
            if show_warning:
                print("Warning: course==-1 appears, previous computation might not be reliable")
            if i == (l-1):
                values[i] = values[i-1]
            else:
                if values[i+1] == -1:
                    return None
                values[i] = values[i+1]
    return values

def get_interpolated_speed_xy(res, hz=15):     
    def vec(speed, course):
        t = math.radians(course)
        return np.array([math.sin(t)*speed, math.cos(t)*speed])
    
    course = res['course']
    speed0 = res['speed']
    # first convert to speed vecs
    l=len(course)
    speed = np.zeros((l, 2), dtype = np.float32)
    for i in range(l):
        # interpolate when the number of missing speed is small
        speed0 = fill_missing_speeds_and_courses(speed0, False)
        course = fill_missing_speeds_and_courses(course, True)
        if (speed0 is None) or (course is None):
            return None

        speed[i,:] = vec(speed0[i], course[i])

    tot_ms = res['endTime'] - res['startTime']
    # total number of output
    nout = tot_ms * hz // 1000
    out = np.zeros((nout, 2), dtype=np.float32)
    
    # if time is t second, there should be t+1 points
    last_start = 0
    ts = res['timestamp']
    for i in range(nout):
        # convert to ms timestamp
        timenow = i * 1000.0 / hz + res['startTime']  
        
        while (last_start+1 < len(ts)) and (ts[last_start+1] < timenow):
            last_start += 1
 
        if last_start+1 == len(ts):                    
            out[i, :] = speed[last_start, :]           
        elif timenow <= ts[0]:
            out[i, :] = speed[0, :]
        else:
            time1 = timenow - ts[last_start]
            time2 = ts[last_start+1] - timenow
            r1 = time2 / (time1 + time2)
            r2 = time1 / (time1 + time2)
            inter = r1*speed[last_start, :] + r2*speed[last_start+1, :]
            out[i, :] = inter
    return out

def get_interpolated_speed(json_path, video_filename, hz):
    res = get_gps(json_path, video_filename)
    if res is None:
        return None
    out = get_interpolated_speed_xy(res, hz)
    return out

def gps_to_loc(speed, hz):
    l = len(speed)
    x_start = 0
    y_start = 0
    x = [x_start]
    y = [y_start]
    dtime = 1.0/hz
    for i in range(l-1):
        x.append(x[i]+dtime*speed[i][0])
        y.append(y[i]+dtime*speed[i][1])
        
    return np.array(x),np.array(y)

def fit_plane(T):
    l = len(T)
    T = np.array(T).T
    plane = np.linalg.svd(T)
    #print(plane[0])
    plane = plane[0][:,2]
    print(plane)
    
    a = np.ones([1,l]) 
    new_T = np.concatenate((T,a), axis = 0)
    new_plane = np.linalg.svd(new_T)
    new_plane = new_plane[0][:,2]
    print(new_plane)
    return plane

def compute_R(vec_1, vec_2):
    
    assert(np.linalg.norm(vec_1) < 1.01 and np.linalg.norm(vec_1) > 0.99)
    assert(np.linalg.norm(vec_2) < 1.01 and np.linalg.norm(vec_2) > 0.99)
    cross_product = np.cross(vec_1, vec_2)
    inner_product = np.inner(vec_1, vec_2)
    sine = np.linalg.norm(cross_product)
    cos  = inner_product
    v = cross_product
    v_cross = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
   
    R = np.eye(3) + v_cross + v_cross.dot(v_cross)*(1-cos)/(sine**2)
    return R

def camera_motion(camera_path,loc):
    R = []
    T = []
    Name = []
    l = loc.shape[1]
    with open(camera_path) as f:
        content = json.load(f)
    content = content[0]['shots']
    #print(content)
    for i in range(l):
        key = str(i+1).zfill(4)+'.jpg'
        if content.has_key(key):
            r = cv2.Rodrigues(np.array(content[key]['rotation']))
            t = np.array(content[key]['translation'])
            R.append(r[0])
            T.append(t)
            Name.append(key)
    return R, T, Name

def camera_pose(R,T):
    P = []
    l = len(T)
    for i in range(l):
        pose = -np.array(R[i]).T.dot(T[i])
        P.append(pose)
    return P

def evaluate_absolute(loc_gps, loc_rec, image_list_file):
    index = []
    error = []
    error_r = []
    with open(image_list_file) as f:
        content = f.readlines()
        for item in content:
            imname = content.split('/')[1]
            imnum = imname.split('.')[0]
            index.append(int(imnum))
    
    
    for i in range(len(loc_rec)):
        error.append(np.linalg.norm(loc_rec[i]-loc_gps[index[i]-1]))
    e = np.mean(error)
    return e

def compute_scale(T, Name, loc):
    scale = []
    loc = loc.T
    for i in range(0,len(T)-1):
        name_1 = int(Name[i].split('.')[0])
        name_2 = int(Name[i+1].split('.')[0])
      
        dist_cam = T[i+1] - T[i]
        dist_loc = loc[name_2 - 1] - loc[name_1 - 1] # gps
       
        scale.append(np.linalg.norm(dist_loc)/np.linalg.norm(dist_cam))
    scale = np.array(scale)
    index = np.argsort(scale)
    scale = scale[index[5:-5]]
    return scale

def align(loc, P, scale):
    loc_scale = loc*(1/scale)
    loc_scale = loc_scale.T
    loc_scale = loc_scale - loc_scale[0,:]
    T_ = np.array(P).T
    T_ = T_.T
    T_ = T_ - T_[0,:]
    return loc_scale, T_

def rotate_gps(loc, T_, Name):
    l = T_.shape[0]
    loc_scale = loc
    n = 50
    for i in range(1):
        name_1 = int(Name[i].split('.')[0])
        name_2 = int(Name[i+n].split('.')[0])
        
        vec_target = T_[i+n] - T_[i]
        vec_source = loc_scale[name_2 - 1] - loc_scale[name_1 - 1]
        vec_target = vec_target[0:2]/np.linalg.norm(vec_target[0:2])
        vec_source = vec_source[0:2]/np.linalg.norm(vec_source[0:2])
        #vec_target = np.flipud(vec_target) #Should delete
        cos = np.inner(vec_target, vec_source)
        cross_product = np.cross(vec_target, vec_source)
        sine = np.linalg.norm(cross_product)
        R_2D = np.array([[cos,-sine],[sine, cos]])
        new_pos = R_2D.dot(loc_scale[:,0:2].T)
        loc_scale_new = loc_scale
        loc_scale_new[:,0:2] = new_pos.T
    return loc_scale_new

def evaluation_rel(loc_1, loc_2, image_list_file):
    loc_gps = loc_1[:,:2]
    loc_rec = np.array(loc_2)[:,:2]
    index = []
    error_r = []
    angle = []
    a_1 = []
    a_2 = []
    t_rec = []
    t_gps = []
    error_trans = []
    
    with open(image_list_file) as f:
        content = f.readlines()
        for item in content:
            imname = item.split('/')[1]
            imnum = imname.split('.')[0]
            index.append(int(imnum)-1)
    
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)
    
    def angle_between(v1, v2):
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        
        angle = np.math.atan2(np.linalg.det([v1_u,v2_u]),np.clip(np.dot(v1_u,v2_u), -1.0, 1.0))
        '''
        if np.degrees(angle) <=-90 :
            angle = 180 + np.degrees(angle)
        elif np.degrees(angle) >= 90:
            angle = 180 - np.degrees(angle) '''
            
        return np.degrees([angle])[0]
    
    orig_list = range(0,len(loc_rec)-16,15)
    for i in orig_list:
        
        for j in range(i+15,len(loc_rec),15):
            ## rotational ##
            if i == 0:
                start_rec = loc_rec[j] - loc_rec[i]
                start_gps = loc_gps[index[j]] - loc_gps[index[i]]
                
            elif j == i+15:
                start_rec = vec_rec
                start_gps = vec_gps

            vec_rec = loc_rec[j] - loc_rec[i]
            vec_gps = loc_gps[index[j]] - loc_gps[index[i]]
            
            #trans_rec = np.linalg.norm(vec_rec)
            #trans_gps = np.linalg.norm(vec_gps)
            
            
            
            angle_rec = angle_between(vec_rec, start_rec)
            angle_gps = angle_between(vec_gps, start_gps)
            if angle_rec < -90 or angle_rec > 90:
                
                orig_list.remove(j)
                
                continue
            else:
                #t_rec.append(trans_rec)
                #t_gps.append(trans_gps)
                break
        a_1.append(angle_rec)
        a_2.append(angle_gps)
        
        angle_error = angle_rec - angle_gps
        angle.append(angle_error)
    
    for i in range(len(loc_rec)):
        
        for j in range(i+1,len(loc_rec)):
            ## rotational ##
            if i == 0:
                start_rec = loc_rec[j] - loc_rec[i]
                start_gps = loc_gps[index[j]] - loc_gps[index[i]]
                
            elif j == i+1:
                start_rec = vec_rec
                start_gps = vec_gps

            vec_rec = loc_rec[j] - loc_rec[i]
            vec_gps = loc_gps[index[j]] - loc_gps[index[i]]
            
            trans_rec = np.linalg.norm(vec_rec)
            trans_gps = np.linalg.norm(vec_gps)
            
            t_rec.append(trans_rec)
            t_gps.append(trans_gps)
            break
        
    for i in range(len(t_rec)-2):
        rate_rec = t_rec[i]/(t_rec[i]+t_rec[i+1]+t_rec[i+2]) + 1e-5
        rate_gps = t_gps[i]/((t_gps[i]+t_gps[i+1]+t_gps[i+2])+1e-5)  + 1e-5
        if np.isnan(np.abs(rate_rec-rate_gps)/rate_gps):
            print(rate_rec, 'rec')
            print(rate_gps, 'gps')
        error_trans.append(np.abs(rate_rec-rate_gps)/rate_gps)
    
    return angle, a_1, a_2, error_trans

def parse_dso(path_to_dso):
    pos_list = []
    with open(path_to_dso) as f:
        content = f.readlines()
        for item in content:
            if item.strip() == "error":
                return None
            i_list = item.split()
            pos = i_list[1:4]
            pos = np.array(pos, dtype=np.float32)
            #print(pos)
            pos_list.append(pos)
    
    return pos_list

def parse_orb(path_to_orb):
    pos_list = []
    with open(path_to_orb) as f:
        content = f.readlines()
        for item in content:
            if item.strip() == "error":
                return None
            i_list = item.split()
            pos = i_list[1:4]
            pos = np.array(pos, dtype=np.float32)
            pos_list.append(pos)
    pln = fit_plane(pos_list)
    R_proj = compute_R(pln,np.array([0,0,1]))
    T_ = R_proj.dot(np.array(pos_list).T)
    return T_.T

def ground_truth(path_to_gps, path_to_video):
    content = get_gps(path_to_gps, path_to_video)
    content['course']=fill_missing_speeds_and_courses(content['course'],1)
    out = get_interpolated_speed(path_to_gps, path_to_video, 30)
    x,y = gps_to_loc(out, 30)
    loc = np.array([x,y])
    Z = np.expand_dims(np.zeros_like(x),0)
    loc = np.concatenate((loc,Z),axis=0)
    return loc

thresh_1 = 0.1
thresh_2 = 1.0
thresh_3 = 1.0

# return whether this example succeed
def eval_one(folder, P):
    # folder is a single path to a folder
    path_to_gps = os.path.join(folder, 'ride.json')
    videos = [each for each in os.listdir(folder) if each.endswith('.mov')]
    path_to_video = videos[0]
    loc = ground_truth(path_to_gps, path_to_video)

    angle, a1, a2, e = evaluation_rel(loc.T, P, image_list_file=os.path.join(folder,'image_list.txt'))
    print(angle)

    v1 = np.sum(angle)
    v2 = np.sum(np.abs(angle))
    v3 = np.sum(np.abs(e))
    print("angle total ", v1, ", angle abs ", v2, ", trans ", v3)
    if v1 >= thresh_1 or v2 >= thresh_2 or v3 > thresh_3:
        return False
    else:
        return True

if __name__ == '__main__':
    program = sys.argv[1]
    folder = sys.argv[2]
    masked = sys.argv[3]
    mask_str = "_mask" if masked.lower() == "true" else ""

    if program == "dso":
        P = parse_dso(folder + "/" + "dso" + mask_str + ".txt")
    elif program == "orb":
        P = parse_orb(folder + "/" + "orb" + mask_str + ".txt")
    elif program == "opensfm":
        folder = folder.rstrip("/")
        if mask_str == "":
            folder += "_nomask"

        R, T, Name = camera_motion(folder+"/reconstruction.json", np.zeros(1, 2000))
        P = camera_pose(R, T)

    if P is None:
        succeed = False
    else:
        succeed = eval_one(folder, P)
    print(succeed)



    
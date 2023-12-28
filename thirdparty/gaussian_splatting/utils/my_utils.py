import numpy as np 



def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w




def posetow2c_matrcs(poses):
    tmp = inversestep4(inversestep3(inversestep2(inversestep1(poses))))
    N = tmp.shape[0]
    ret = []
    for i in range(N):
        ret.append(tmp[i])
    return ret


def getRTfromPose(w2c_mats):
    for m in w2c_mats:
        R = m[:3, :3]
        t = m[:3, 3]
        #print(R, t)

def tolist(w2c_mats):
    return w2c_mats.tolist()
def inversestep4(c2w_mats):
    return np.linalg.inv(c2w_mats)
def inversestep3(newposes):
    tmp = newposes.transpose([2, 0, 1]) # 20, 3, 4 
    N, _, __ = tmp.shape
    zeros = np.zeros((N, 1, 4))
    zeros[:, 0, 3] = 1
    c2w_mats = np.concatenate([tmp, zeros], axis=1)
    return c2w_mats

def inversestep2(newposes):
    return newposes[:,0:4, :]
def inversestep1(newposes):
    poses = np.concatenate([newposes[:, 1:2, :], newposes[:, 0:1, :], -newposes[:, 2:3, :],  newposes[:, 3:4, :],  newposes[:, 4:5, :]], axis=1)
    return poses





def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec



def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
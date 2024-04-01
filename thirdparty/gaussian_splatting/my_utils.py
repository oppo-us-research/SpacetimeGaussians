import numpy as np


def pose_to_w2c_matrixes(poses):
    tmp = inverse_step4(inverse_step3(inverse_step2(inverse_step1(poses))))
    N = tmp.shape[0]
    ret = []
    for i in range(N):
        ret.append(tmp[i])
    return ret


def getRTfromPose(w2c_mats):
    for m in w2c_mats:
        R = m[:3, :3]
        t = m[:3, 3]
        # print(R, t)


def tolist(w2c_mats):
    return w2c_mats.tolist()


def inverse_step4(c2w_mats):
    return np.linalg.inv(c2w_mats)


def inverse_step3(new_poses):
    tmp = new_poses.transpose([2, 0, 1])  # 20, 3, 4
    N, _, __ = tmp.shape
    zeros = np.zeros((N, 1, 4))
    zeros[:, 0, 3] = 1
    c2w_mats = np.concatenate([tmp, zeros], axis=1)
    return c2w_mats


def inverse_step2(new_poses):
    return new_poses[:, 0:4, :]


def inverse_step1(new_poses):
    poses = np.concatenate(
        [
            new_poses[:, 1:2, :],
            new_poses[:, 0:1, :],
            -new_poses[:, 2:3, :],
            new_poses[:, 3:4, :],
            new_poses[:, 4:5, :],
        ],
        axis=1,
    )
    return poses


def rot_mat_2_qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eig_vals, eig_vecs = np.linalg.eigh(K)
    qvec = eig_vecs[[3, 0, 1, 2], np.argmax(eig_vals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def qvec_2_rot_mat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

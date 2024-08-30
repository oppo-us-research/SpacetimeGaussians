import sys
sys.path.append(".")
from thirdparty.colmap.pre_colmap import *
from pathlib import Path

def write_colmap(path, cameras, offset=0):
    projectfolder = path / f"colmap_{offset}"
    manualfolder = projectfolder / "manual"
    manualfolder.mkdir(exist_ok=True)

    savetxt = manualfolder / "images.txt"
    savecamera = manualfolder / "cameras.txt"
    savepoints = manualfolder / "points3D.txt"

    imagetxtlist = []
    cameratxtlist = []

    db_file = projectfolder / "input.db"
    if db_file.exists():
        db_file.unlink()

    db = COLMAPDatabase.connect(db_file)

    db.create_tables()


    for cam in cameras:
        id = cam['id']
        filename = cam['filename']

        # intrinsics
        w = cam['w']
        h = cam['h']
        fx = cam['fx']
        fy = cam['fy']
        cx = cam['cx']
        cy = cam['cy']

        # extrinsics
        colmapQ = cam['q']
        T = cam['t']

        # check that cx is almost w /2, idem for cy
        assert abs(cx - w / 2) / cx < 0.10, f"cx is not close to w/2: {cx}, w: {w}"
        assert abs(cy - h / 2) / cy < 0.10, f"cy is not close to h/2: {cy}, h: {h}"

        line = f"{id} " + " ".join(map(str, colmapQ)) + " " + " ".join(map(str, T)) + f" {id} {filename}\n"
        imagetxtlist.append(line)
        imagetxtlist.append("\n")

        params = np.array((fx , fy, cx, cy,))

        camera_id = db.add_camera(1, w, h, params)
        cameraline = f"{id} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n"
        cameratxtlist.append(cameraline)
        image_id = db.add_image(filename, camera_id,  prior_q=colmapQ, prior_t=T, image_id=id)
        db.commit()
    db.close()

    savetxt.write_text("".join(imagetxtlist))
    savecamera.write_text("".join(cameratxtlist))
    savepoints.write_text("")  # Creating an empty points3D.txt file
    
import mujoco
import mujoco_viewer
import sys 
sys.path.append('../../../')


if __name__=="__main__":
    model = mujoco.MjModel.from_xml_path("./../../../assets/franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    data.qpos = model.keyframe('home').qpos
    mujoco.mj_forward(model, data)
    while True: 
        mujoco.mj_step(model, data)
        viewer.render()
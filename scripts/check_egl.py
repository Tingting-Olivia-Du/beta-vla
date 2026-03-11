#!/usr/bin/env python3
"""Check if EGL/OSMesa is available for headless rendering (LIBERO/robosuite)."""
import os
import sys

def check_egl():
    """Try EGL rendering."""
    os.environ["MUJOCO_GL"] = "egl"
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>")
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, 64, 64)
        r.update_scene(d)
        r.render()
        print("[OK] EGL: GPU-accelerated headless rendering works")
        return True
    except Exception as e:
        print(f"[FAIL] EGL: {e}")
        return False

def check_osmesa():
    """Try OSMesa (CPU) rendering."""
    os.environ["MUJOCO_GL"] = "osmesa"
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>")
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, 64, 64)
        r.update_scene(d)
        r.render()
        print("[OK] OSMesa: CPU software rendering works")
        return True
    except Exception as e:
        print(f"[FAIL] OSMesa: {e}")
        return False

def main():
    print("Checking headless rendering backends for LIBERO/robosuite...")
    print()
    egl_ok = check_egl()
    print()
    osmesa_ok = check_osmesa()
    print()
    if egl_ok:
        print("Recommendation: Use MUJOCO_GL=egl (default, GPU-accelerated)")
    elif osmesa_ok:
        print("Recommendation: Use MUJOCO_GL=osmesa (CPU fallback, slower)")
    else:
        print("Recommendation: Install osmesa: apt install libosmesa6-dev")
        sys.exit(1)

if __name__ == "__main__":
    main()

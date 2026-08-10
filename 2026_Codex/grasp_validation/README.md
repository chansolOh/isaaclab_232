# Output Grasp Validation

## Run

```bash
cd /home/uon/ochansol/isaaclab_232
uv run python 2026_Codex/grasp_validation/grasp_bbox_viewer_validation.py
```

Edit `self.root_path` near the top of `ImageGraspViewer.__init__` when a
different dataset root is required.

## Viewer workflow

1. Select Environment, Section, and Platform, then click `Load Images`.
2. Select an image(scene) and BBox(grasp).
3. Click `Load to Sim` once.
4. Use the BBox slider or `BBox Navigation` buttons to switch the replayed
   grasp. The same scene and gripper reuse the running Isaac Lab environment.
5. Moving to another image(scene), or to another gripper model, restarts the
   replay process and rebuilds the stage.

The selected episode repeats until another grasp is selected or the viewer is
closed. Closing the viewer also terminates the Isaac Lab replay process.

## Simulator debug colors

- Yellow cross: output `target_points`
- Magenta line: hand START to END base trajectory
- Red/green/blue: pose X/Y/Z axes
- Cyan: matched pre_grasp 3D grasp bbox

The cyan bbox is shown only when the selected output record can be matched to
a pre_grasp record that contains `grasp_bbox`. No 3D bbox is invented when the
source pre_grasp lacks that field.

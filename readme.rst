
.. Keep this document short & concise,
   linking to external resources instead of including content in-line.
   See 'release/text/readme.html' for the end user read-me.


Blender implementation of "Tangent-Space Optimization for Interactive Animation Control"
================================================================================================
This is a fork of the Blender repo. I implemented the interactive motion path editing gizmo in this repo. The executable for Windows10 can be found under `Release <https://github.com/JiahuiCai/Blender_Interactive_Motion_Path/releases>`__.

- The core idea comes from the Disney Research paper, `Tangent-Space Optimization for Interactive Animation Control <https://studios.disneyresearch.com/2019/07/12/tangent-space-optimization-of-controls-for-character-animation/>`__

- The code is presented for reference only. The tool is not ready for production due to some limitations mentioned below, specifically quaternions do not work at all. 

- Feel free to ask me any questions if the code is confusing as I had to hack a bit for specific gizmo behaviors.

.. image:: https://github.com/JiahuiCai/FileStorage/blob/master/AnimationTest.gif?raw=true
   :width: 49%
.. image::  https://github.com/JiahuiCai/FileStorage/blob/master/motion_edit_with_constraints.gif?raw=true
   :width: 49%
.. image:: https://github.com/JiahuiCai/FileStorage/blob/master/longchain.gif?raw=true
   :width: 49%

Features:
---------
- Motion path of all selected bones can be visualized in the viewport directly.
- Left click and drag to alter the motion path at **current** frame time.
    - If there is a keyframe at the current frame time, the keyframe is adjusted.
    - Otherwise the tangents of the fcurves are adjusted (Refer to the paper above)
    - **It is important to understand that it does not matter which frame-dot you click on the motion path, you are still changing the pose of the character at the CURRENT frame shown in the time line. This is intentional as it helps avoid misclick when multiple frame-dots on the motion path are bunched up. Read the next feature to find out how to change current frame time directly in the viewport**.
    - Useful for giving FK-like interpolation to IK controllers or IK-like interpolation for FK rigs. You don't need IK/FK switch anymore.
    - No additional keyframes are added, only the tangents are modified, which gives nice propotional editing behavior by default.

.. image:: https://raw.githubusercontent.com/JiahuiCai/FileStorage/master/LeftMouseButtonClickAndDrag.gif
   :width: 49%
   :align: center

- Alt + Left mouse button, click to change frame time.

.. image:: https://raw.githubusercontent.com/JiahuiCai/FileStorage/master/AltLeftMouseButtonClick.gif
   :width: 49%
   :align: center

- Ctrl + Left mouse button, click to pin a point on the motion path. (Useful for doing heel roll control on a FK rig);
- Ctrl + Left mouse button, drag to pin the entire motion path.

.. image:: https://raw.githubusercontent.com/JiahuiCai/FileStorage/master/CtrlLeftMouseButton.gif
   :width: 49%
   :align: center

- Shift + Left mouse button, drag on a segment between two keyframes to interpolate linearly between two locations in world space. (works for both IK/FK controls)

.. image:: https://github.com/JiahuiCai/FileStorage/blob/master/ShiftLeftMouseButton.gif?raw=true
   :width: 49%
.. image::  https://github.com/JiahuiCai/FileStorage/blob/master/ShiftLeftMouseButton2.gif?raw=true 
   :width: 49%

- In the "N" panel, there is a "range" option to specify how many keyframes around the current frame to display for all motion paths.

.. image:: https://raw.githubusercontent.com/JiahuiCai/FileStorage/master/range.gif
   :width: 49%
   :align: center


- In the "Bone" properties panel, right below IK setting there is the Motion Curve setting for the selected bone.
    - You can choose to disable visualization of the motion path gizmo for the bone head/tail. (Useful for controls that are only keyed on locations(Head only))
    - You can choose to filter the fcurves involved in the path adjustment. (Useful for joints with limited DOF, such as elbows and knees)
    - You can choose the number of bones along a bone chain that are involved in the path adjustment. (For example: you can set it to 2 for arms and legs to get regular IK behaviors, and set it to 0 for full upper/lower body IK) 

.. image:: https://raw.githubusercontent.com/JiahuiCai/FileStorage/master/bone_settings.gif
   :width: 49%
   :align: center

Limitations:
------------
- Only works on location fcurves and Euler angle fcurves, quaternions and blender bone constraints are not supported.
- Interpolation adjustments only works for bezier curve interpolation mode.
- The tool is designed for global poses, meaning you will need to key the entire character even if you only change the arms or do any minor adjustments. This also means that shifting keys to create overlap motions is not supported. You need to align all keyframes. 
- Object mode transforms are not supported. Make sure your object transfrom is zeroed out.
- Rotational adjustment feature is limited, twist motion along the bone's local-Y axis cannot be accomplished using the tools' interface. But you could pin the bones first, use the blender rotation tool to rotate and then click on the motion path to allow the solver to recover the pinned locations.
- My original intention is to have this motion trail/pin&drag based workflow replace the need for a complex rig. But unfortunately the pin&drag workflow is equally as tedious as working with a complex FK/IK switching rig. I would recommend you use this tool with a basic IK rig since at the very least you can achieve both IK&FK behaviors with a simple IK rig using this tool, eliminating the need for IK/FK switching.

Files:
------------
For implementation details, please refer to the following files:

- `source/blender/editors/armature/pose_anim_motion_curve.cc <https://github.com/JiahuiCai/Blender_Interactive_Motion_Path/blob/interactive_motion_path/source/blender/editors/armature/pose_anim_motion_curve.cc>`__

Minor changes in:

- release/scripts/startup/bl_ui/properties_data_bone.py 

- release/scripts/startup/bl_ui/space_toolsystem_toolbar.py

- source/blender/editors/armature/CMakeLists.txt

- source/blender/editors/armature/armature_intern.h

- source/blender/editors/armature/armature_ops.c

Blender
=======

Blender is the free and open source 3D creation suite.
It supports the entirety of the 3D pipeline-modeling, rigging, animation, simulation, rendering, compositing,
motion tracking and video editing.

.. figure:: https://code.blender.org/wp-content/uploads/2018/12/springrg.jpg
   :scale: 50 %
   :align: center


Project Pages
-------------

- `Main Website <http://www.blender.org>`__
- `Reference Manual <https://docs.blender.org/manual/en/latest/index.html>`__
- `User Community <https://www.blender.org/community/>`__

Development
-----------

- `Build Instructions <https://wiki.blender.org/wiki/Building_Blender>`__
- `Code Review & Bug Tracker <https://developer.blender.org>`__
- `Developer Forum <https://devtalk.blender.org>`__
- `Developer Documentation <https://wiki.blender.org>`__


License
-------

Blender as a whole is licensed under the GNU Public License, Version 3.
Individual files may have a different, but compatible license.

See `blender.org/about/license <https://www.blender.org/about/license>`__ for details.

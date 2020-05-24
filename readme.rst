
.. Keep this document short & concise,
   linking to external resources instead of including content in-line.
   See 'release/text/readme.html' for the end user read-me.


Jiahui(Jack) Cai's Note: 
-----------------
This is a fork of the Blender repo. I implemented the interactive motion path editing gizmo in this repo. Feel free to ask me any questions if the code is confusing because I had to hack a bit for specific gizmo behaviors.

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

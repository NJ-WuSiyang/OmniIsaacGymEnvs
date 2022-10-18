# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional
import math
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysxSchema

class UR10(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "ur10",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]
        """

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([0.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        if self._usd_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "base_link/shoulder_pan_joint",
            "shoulder_link/shoulder_lift_joint",
            "upper_arm_link/elbow_joint",
            "forearm_link/wrist_1_joint",
            "wrist_1_link/wrist_2_joint",
            "wrist_2_link/wrist_3_joint",
        ]

        drive_type = ["angular"] * len(dof_paths)
        stiffness = [349.06586] * len(dof_paths)
        damping = [34.90659] * len(dof_paths)
        max_force = [1979.99988, 1979.99988, 899.99994, 336.0, 336.0, 336.0]

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=0,
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )

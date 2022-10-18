# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.ur5e import UR5e
from omniisaacgymenvs.robots.articulations.ur10 import UR10
from omniisaacgymenvs.robots.articulations.my_franka import MyFranka
from omniisaacgymenvs.robots.articulations.views.ur5e_view import UR5eView
from omniisaacgymenvs.robots.articulations.views.ur10_view import UR10View
from omniisaacgymenvs.robots.articulations.views.my_franka_view import MyFrankaView

from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import *

from omni.isaac.cloner import Cloner

import numpy as np
import torch
import math

from pxr import Usd, UsdGeom


class PlaygroundTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self._use_ur5e = self._task_cfg["env"]["use_ur5e"]
        self._use_ur10 = self._task_cfg["env"]["use_ur10"]
        self._use_myfranka = self._task_cfg["env"]["use_myfranka"]

        self.distX_offset = 0.04
        self.dt = 1/60.

        self._num_observations = 1
        self._num_actions = 6

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:

        if self._use_ur5e:
            self.get_ur5e()
        elif self._use_ur10:
            self.get_ur10()
        elif self._use_myfranka:
            self.get_myfranka()

        super().set_up_scene(scene)

        if self._use_ur5e:
            self._ur5es = UR5eView(prim_paths_expr="/World/envs/.*/ur5e", name="ur5e_view")
            scene.add(self._ur5es)
        elif self._use_ur10:
            self._ur10s = UR10View(prim_paths_expr="/World/envs/.*/ur10", name="ur10_view")
            scene.add(self._ur10s)
        elif self._use_myfranka:
            self._myfrankas = MyFrankaView(prim_paths_expr="/World/envs/.*/myfranka", name="myfranka_view")
            scene.add(self._myfrankas)

        self.init_data()
        return

    def get_myfranka(self):
        myfranka = MyFranka(prim_path=self.default_zero_env_path + "/myfranka", name="myfranka",
                            usd_path=self._task_cfg["sim"]["myfranka_usd_path"])
        self._sim_config.apply_articulation_settings("myfranka", get_prim_at_path(myfranka.prim_path), self._sim_config.parse_actor_config("myfranka"))

    def get_ur10(self):
        ur10 = UR10(prim_path=self.default_zero_env_path + "/ur10", name="ur10",
                    usd_path=self._task_cfg["sim"]["ur10_usd_path"])
        self._sim_config.apply_articulation_settings("ur10", get_prim_at_path(ur10.prim_path), self._sim_config.parse_actor_config("ur10"))

    def get_ur5e(self):
        ur5e = UR5e(prim_path=self.default_zero_env_path + "/ur5e", name="ur5e",
                    usd_path=self._task_cfg["sim"]["ur5e_usd_path"])
        self._sim_config.apply_articulation_settings("ur5e", get_prim_at_path(ur5e.prim_path), self._sim_config.parse_actor_config("ur5e"))

    def init_data(self) -> None:
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)
        if self._use_ur5e:
            self.ur5e_default_dof_pos = torch.zeros(6, device=self._device)
        elif self._use_ur10:
            self.ur10_default_dof_pos = torch.zeros(6, device=self._device)
        elif self._myfrankas:
            self.myfranka_default_dof_pos = torch.tensor([math.degrees(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]] + [0.02, 0.02], device=self._device)

    def get_observations(self) -> dict:
        if self._use_ur5e:
            observations = {
                self._ur5es.name: {
                    "obs_buf": self._ur5es.get_joint_positions(clone=False)
                }
            }
        elif self._use_ur10:
            observations = {
                self._ur10s.name: {
                    "obs_buf": self._ur10s.get_joint_positions(clone=False)
                }
            }
        elif self._use_myfranka:
            observations = {
                self._myfrankas.name: {
                    "obs_buf": self._myfrankas.get_joint_positions(clone=False)
                }
            }
        return observations

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)

        if self._use_ur5e:
            targets = self.ur5e_dof_targets + self.dt * self.actions
            self.ur5e_dof_targets[:] = targets
            env_ids_int32 = torch.arange(self._ur5es.count, dtype=torch.int32, device=self._device)

            self._ur5es.set_joint_position_targets(self.ur5e_dof_targets, indices=env_ids_int32)
        elif self._use_ur10:
            targets = self.ur10_dof_targets + self.dt * self.actions
            self.ur10_dof_targets[:] = targets
            env_ids_int32 = torch.arange(self._ur10s.count, dtype=torch.int32, device=self._device)

            self._ur10s.set_joint_position_targets(self.ur10_dof_targets, indices=env_ids_int32)
        elif self._use_myfranka:
            targets = self.myfranka_dof_targets + self.dt * self.actions
            self.myfranka_dof_targets[:] = targets
            env_ids_int32 = torch.arange(self._myfrankas.count, dtype=torch.int32, device=self._device)

            self._myfrankas.set_joint_position_targets(self.myfranka_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        if self._use_ur5e:
            pos = self.ur5e_default_dof_pos.unsqueeze(0)
            dof_pos = torch.zeros((num_indices, self._ur5es.num_dof), device=self._device)
            dof_vel = torch.zeros((num_indices, self._ur5es.num_dof), device=self._device)
            dof_pos[:, :] = pos
            self.ur5e_dof_targets[env_ids, :] = pos

            self._ur5es.set_joint_position_targets(self.ur5e_dof_targets[env_ids], indices=indices)
            self._ur5es.set_joint_positions(dof_pos, indices=indices)
            self._ur5es.set_joint_velocities(dof_vel, indices=indices)
        elif self._use_ur10:
            pos = self.ur10_default_dof_pos.unsqueeze(0)
            dof_pos = torch.zeros((num_indices, self._ur10s.num_dof), device=self._device)
            dof_vel = torch.zeros((num_indices, self._ur10s.num_dof), device=self._device)
            dof_pos[:, :] = pos
            self.ur10_dof_targets[env_ids, :] = pos

            self._ur10s.set_joint_position_targets(self.ur10_dof_targets[env_ids], indices=indices)
            self._ur10s.set_joint_positions(dof_pos, indices=indices)
            self._ur10s.set_joint_velocities(dof_vel, indices=indices)
        elif self._use_myfranka:
            pos = self.myfranka_default_dof_pos.unsqueeze(0)
            dof_pos = torch.zeros((num_indices, self._myfrankas.num_dof), device=self._device)
            dof_vel = torch.zeros((num_indices, self._myfrankas.num_dof), device=self._device)
            dof_pos[:, :] = pos
            self.myfranka_dof_targets[env_ids, :] = pos

            self._myfrankas.set_joint_position_targets(self.myfranka_dof_targets[env_ids], indices=indices)
            self._myfrankas.set_joint_positions(dof_pos, indices=indices)
            self._myfrankas.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        if self._use_ur5e:
            self.num_ur5e_dofs = self._ur5es.num_dof
            self.ur5e_dof_targets = torch.zeros(
                (self._num_envs, self.num_ur5e_dofs), dtype=torch.float, device=self._device
            )
        elif self._use_ur10:
            self.num_ur10_dofs = self._ur10s.num_dof
            self.ur10_dof_targets = torch.zeros(
                (self._num_envs, self.num_ur10_dofs), dtype=torch.float, device=self._device
            )
        elif self._use_myfranka:
            self.num_myfranka_dofs = self._myfrankas.num_dof
            self.myfranka_dof_targets = torch.zeros(
                (self._num_envs, self.num_myfranka_dofs), dtype=torch.float, device=self._device
            )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = 0

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

In robots/articulations/shadow_hand.py:
```python
def set_shadow_hand_properties(self, stage, shadow_hand_prim):
    for link_prim in shadow_hand_prim.GetChildren():
        if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
            rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
            rb.GetDisableGravityAttr().Set(True)
            rb.GetRetainAccelerationsAttr().Set(True)
```


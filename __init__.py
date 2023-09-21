from . import dynthres_comfyui

NODE_CLASS_MAPPINGS = {
    "DynamicThresholdingSimple": dynthres_comfyui.DynamicThresholdingSimpleComfyNode,
    "DynamicThresholdingFull": dynthres_comfyui.DynamicThresholdingComfyNode,
}

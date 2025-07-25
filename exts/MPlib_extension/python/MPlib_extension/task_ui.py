import os
import omni.ext
import asyncio
import omni.ui as ui

from MPlib_extension.pickplace import MyTaskRunner
from isaacsim.gui.components.ui_utils import btn_builder
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate


class MPlibExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "Franka Pick-and-Place (WIP for MPLib)"
        self.category = "[!] Custom Examples"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Franka Pick-and-Place Task",
            "doc_link": "",
            "overview": "Spawns a Franka arm and runs a pick-and-place demo using RRT motion planning [WIP for using MPlib in the future].",
            "sample": MyTaskRunner(),
        }

        ui_handle = MPlibUI(**ui_kwargs)

        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)
        print("[DEBUG] MPlibExtension.on_shutdown called")

class MPlibUI(BaseSampleUITemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_extra_frames(self):
        with self.get_extra_frames_handle():
            with ui.CollapsableFrame(title="Task Control", height=0):
                with ui.VStack(spacing=5):
                    btn = btn_builder(
                        label="Run Pick-and-Place",
                        text="Run",
                        tooltip="Pick and place using RRT planner",
                        on_clicked_fn=self._on_pick_and_place,
                    )
                    btn.enabled = True


    def _on_pick_and_place(self):
        asyncio.ensure_future(self.sample._on_follow_target_event_async())
    


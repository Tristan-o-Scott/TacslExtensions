import os
import omni.ext
import asyncio
import omni.ui as ui
from MPlib_extension.pickplace import PickPlaceTaskRunner
from MPlib_extension.massthrow import MassThrowTaskRunner

from isaacsim.gui.components.ui_utils import btn_builder
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate

class MPlibExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.category = "[!] Custom Examples"

        pickplace_ui = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Pick-and-Place Task",
            "doc_link": "",
            "overview": "Spawns a Franka arm and runs a pick-and-place demo using MPlib.",
            "sample": PickPlaceTaskRunner(),
        }

        pickplace_ui_handle = MPlibUI(**pickplace_ui)

        get_browser_instance().register_example(
            name="Franka Pick-and-Place",
            execute_entrypoint=pickplace_ui_handle.build_window,
            ui_hook=pickplace_ui_handle.build_ui,
            category=self.category,
        )

        massthrow_ui = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Throwing Object Task",
            "doc_link": "",
            "overview": "Franka arm throws objects of different masses using MPlib. The throw strength is adjusted based on the mass of the object and the position of the target zone. "
            "The results from the throw may not always land in the target zone, but the results should be somewhat consistent. Inconsistencies may arise due to the physics engine's behavior.",
            "sample": MassThrowTaskRunner(),
        }
        
        massthrow_ui_handle = MassThrowUI(**massthrow_ui)

        get_browser_instance().register_example(
            name="Franka Mass Throw",
            execute_entrypoint=massthrow_ui_handle.build_window,
            ui_hook=massthrow_ui_handle.build_ui,
            category=self.category,
        )

    def on_shutdown(self):
        get_browser_instance().deregister_example(name="Franka Pick-and-Place", category=self.category)
        get_browser_instance().deregister_example(name="Franka Mass Throw", category=self.category)



class MPlibUI(BaseSampleUITemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_extra_frames(self):
        with self.get_extra_frames_handle():
            with ui.CollapsableFrame(title="Task Control", height=0):
                with ui.VStack(spacing=5):
                    btn1 = btn_builder(
                        label="Run Pick-and-Place",
                        text="Run",
                        tooltip="Pick and place using MPlib planner",
                        on_clicked_fn=self._on_pick_and_place,
                    )
                    btn1.enabled = True

                    btn2 = btn_builder(
                        label="Test MPLib",
                        text="Run Test",
                        tooltip="Moves the robot to a test pose using MPLib, checks installation.",
                        on_clicked_fn=self._on_mplib_test,
                    )
                    btn2.enabled = True


    def _on_pick_and_place(self):
        asyncio.ensure_future(self.sample._on_follow_target_event_async())

    def _on_mplib_test(self):
        asyncio.ensure_future(self.sample.test_mplib_plan_and_execute())

class MassThrowUI(BaseSampleUITemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_extra_frames(self):
        with self.get_extra_frames_handle():
            with ui.CollapsableFrame(title="Mass Throw Control", height=0):
                with ui.VStack(spacing=5):
                    btn = btn_builder(
                        label="Run Mass Throw",
                        text="Throw",
                        tooltip="Throw blocks of different masses using MPlib.",
                        on_clicked_fn=self._on_mass_throw
                    )
                    btn.enabled = True

    def _on_mass_throw(self):
        asyncio.ensure_future(self.sample._on_follow_target_event_async()) 
